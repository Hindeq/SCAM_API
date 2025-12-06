import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from scipy.signal import resample, spectrogram, butter, filtfilt
from scipy.stats import iqr, skew, kurtosis
from scipy.fft import fft
import pandas as pd
import math
import logging
import os
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel

# Configuration logging
logging.basicConfig(level=logging.INFO)

# --- PARAMÈTRES ET CLÉS DE BASE DE DONNÉES ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "URL_PAR_DEFAUT_SI_MANQUANTE")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "CLE_PAR_DEFAUT_SI_MANQUANTE")
SEUIL_FC_STRESS = float(os.getenv("SEUIL_FC_STRESS", 90.0))

# --- SEUILS ADAPTÉS AUX SENIORS (60+) ---
SEUIL_FC_LOW = 50.0          # bradycardie préoccupante
SEUIL_FC_HIGH = 100.0        # tachycardie d'alerte
SEUIL_FC_DANGER = 120.0      # critique
SEUIL_SPO2_NORMAL = 94.0
SEUIL_SPO2_LOW = 92.0
SEUIL_SPO2_CRIT = 91.0

# Paramètres IoT/Cloud & SLIM
FS_BVP = 100.0
FS_ACC = 50.0
WINDOW_SEC = 8
ORDER = 3
MIN_HR_HZ = 40 / 60
MAX_HR_HZ = 220 / 60

# Fenêtre historique pour le POC (nombre d'échantillons)
# (Supposons 1 échantillon / seconde pour le POC ; ajuste selon la cadence réelle.)
HISTORY_LEN = 10   # fenêtre glissante courte (ex : 10s)
BUFFER_MAX_LEN = 60  # historique global (1 min)

# Initialisation de Supabase
supabase = None
try:
    if SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL != "URL_PAR_DEFAUT_SI_MANQUANTE":
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("Connexion Supabase établie.")
    else:
        logging.warning("Variables Supabase manquantes. Connexion BD désactivée.")
except Exception as e:
    logging.error(f"Erreur d'initialisation Supabase: {e}")
    supabase = None

# Modèle de données attendu
class SensorData(BaseModel):
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    bpm: float
    spo2: float
    timestamp: Optional[datetime] = None

app = FastAPI()

# --- Buffers globaux pour suivi temporel (POC simple) ---
# Pour un vrai multi-utilisateur on stockerait par device/user id.
fc_history = []      # stocke floats (bpm)
spo2_history = []    # stocke floats (%)
motion_history = []  # stocke int (classes SLIM 0/1/2)
time_history = []    # timestamps

# --- FONCTIONS DE BASE (inchangées / adaptées) ---
def butter_lowpass_filter(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return filtfilt(b, a, data, axis=0)

def compute_jerk(signal, fs):
    return np.diff(signal, axis=0, prepend=signal[0:1]) * fs

def compute_magnitude(signal):
    return np.linalg.norm(signal, axis=1)

def compute_fft(signal, fs):
    N = len(signal)
    fft_vals = np.abs(fft(signal))
    fft_vals = fft_vals[:N // 2]
    freqs = np.fft.fftfreq(N, 1/fs)[:N // 2]
    return fft_vals, freqs

def signal_magnitude_area(signal):
    return np.sum(np.abs(signal)) / len(signal)

def signal_energy(signal):
    return np.sum(signal**2) / len(signal)

def signal_entropy(signal, bins=10):
    signal = np.asarray(signal)
    if signal.size == 0 or np.all(signal == signal[0]):
        return 0.0
    hist, _ = np.histogram(signal, bins=bins, density=False)
    hist = hist[hist > 0]
    hist = hist / np.sum(hist)
    return float(-np.sum(hist * np.log2(hist)))

def weighted_mean_freq(fft_vals, freqs):
    return np.sum(fft_vals * freqs) / np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0

def bands_energy_proxy(fft_vals):
    return np.sum(fft_vals**2) / np.max(fft_vals) if np.max(fft_vals) > 0 else 0.0

def angle_proxy(v1, v2):
    norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_prod == 0:
        return 0.0
    return np.dot(v1, v2) / norm_prod

def arcoeff_proxy():
    return [0.1, 0.05, 0.02, 0.01]

# --- ESTIMATE FC CORRIGÉE (inchangée) ---
def estimate_fc_corrected(bvp_signal, acc_data):
    num_bvp_samples = bvp_signal.size
    window_samples = int(WINDOW_SEC * FS_BVP)
    step_samples = int(WINDOW_SEC * FS_BVP / 4)

    acc_resampled = np.zeros((num_bvp_samples, 3))
    try:
        for i in range(3):
            acc_resampled[:, i] = resample(acc_data[:, i], num_bvp_samples)
    except (IndexError, ValueError):
        return 0.0

    motion_magnitude = np.sqrt(np.sum(acc_resampled**2, axis=1))
    hr_estimates = []

    for start in range(0, num_bvp_samples - window_samples, step_samples):
        end = start + window_samples
        bvp_window = bvp_signal[start:end]
        acc_window = motion_magnitude[start:end]
        if len(bvp_window) != window_samples:
            continue

        f_bvp, _, Pxx_den_bvp = spectrogram(bvp_window, FS_BVP, nperseg=window_samples, noverlap=0, mode='psd')
        f_acc, _, Pxx_den_acc = spectrogram(acc_window, FS_BVP, nperseg=window_samples, noverlap=0, mode='psd')

        acc_hr_indices = np.where((f_acc >= MIN_HR_HZ) & (f_acc <= MAX_HR_HZ))[0]
        # robust: si peu de peaks, on gère
        if acc_hr_indices.size == 0:
            motion_peaks = np.array([])
        else:
            Pxx_acc_flat = Pxx_den_acc.flatten()
            motion_peaks = f_acc[acc_hr_indices][np.argsort(Pxx_acc_flat[acc_hr_indices])[-2:]]

        bvp_hr_indices = np.where((f_bvp >= MIN_HR_HZ) & (f_bvp <= MAX_HR_HZ))[0]
        Pxx_bvp_hr = Pxx_den_bvp[bvp_hr_indices] if bvp_hr_indices.size>0 else np.array([])
        f_bvp_hr = f_bvp[bvp_hr_indices] if bvp_hr_indices.size>0 else np.array([])

        if len(f_bvp_hr) == 0:
            continue

        candidate_hr_freq = None
        max_power = -1
        for idx, freq in enumerate(f_bvp_hr):
            is_motion = any(np.abs(freq - mp) < 0.1 for mp in motion_peaks) if motion_peaks.size>0 else False
            if not is_motion and Pxx_bvp_hr[idx] > max_power:
                max_power = Pxx_bvp_hr[idx]
                candidate_hr_freq = freq

        if candidate_hr_freq is None:
            candidate_hr_freq = f_bvp_hr[np.argmax(Pxx_bvp_hr)]

        hr_estimates.append(candidate_hr_freq * 60)

    if not hr_estimates:
        return 0.0
    return np.mean(hr_estimates)

# --- CALCUL DES 92 FEATURES (SLIM) : inchangé --- 
# (Je conserve ta fonction exactement comme fournie; elle renvoie un vecteur numpy de 92 features)
def calculate_slim_features(acc_raw_400, gyro_raw_400):
    gravity_acc = butter_lowpass_filter(acc_raw_400, 0.3, FS_ACC, ORDER)
    body_acc = acc_raw_400 - gravity_acc

    body_acc_jerk = compute_jerk(body_acc, FS_ACC)
    body_gyro_jerk = compute_jerk(gyro_raw_400, FS_ACC)

    tBodyAccMag = compute_magnitude(body_acc)
    tGravityAccMag = compute_magnitude(gravity_acc)
    tBodyAccJerkMag = compute_magnitude(body_acc_jerk)
    tBodyGyroMag = compute_magnitude(gyro_raw_400)
    tBodyGyroJerkMag = compute_magnitude(body_gyro_jerk)

    signals_3d = {'tBodyAcc': body_acc, 'tGravityAcc': gravity_acc, 'tBodyAccJerk': body_acc_jerk,
                  'tBodyGyro': gyro_raw_400, 'tBodyGyroJerk': body_gyro_jerk}
    signals_1d = {'tBodyAccMag': tBodyAccMag, 'tGravityAccMag': tGravityAccMag, 'tBodyAccJerkMag': tBodyAccJerkMag,
                  'tBodyGyroMag': tBodyGyroMag, 'tBodyGyroJerkMag': tBodyGyroJerkMag}

    calculated_features = {}
    for name, sig in signals_3d.items():
        for i, axis in enumerate(['X', 'Y', 'Z']):
            calculated_features[f'{name}-std()-{axis}'] = np.std(sig[:, i])
            calculated_features[f'{name}-mad()-{axis}'] = np.median(np.abs(sig[:, i] - np.median(sig[:, i])))
            calculated_features[f'{name}-entropy()-{axis}'] = signal_entropy(sig[:, i])
            calculated_features[f'{name}-iqr()-{axis}'] = iqr(sig[:, i])
            calculated_features[f'{name}-energy()-{axis}'] = signal_energy(sig[:, i])
            calculated_features[f'{name}-mean()-{axis}'] = np.mean(sig[:, i])
            calculated_features[f'{name}-max()-{axis}'] = np.max(sig[:, i])
            for j, val in enumerate(arcoeff_proxy()):
                calculated_features[f'{name}-arCoeff()-{axis},{j+1}'] = val

    for name, sig in signals_1d.items():
        calculated_features[f'{name}-mean()'] = np.mean(sig)
        calculated_features[f'{name}-std()'] = np.std(sig)
        calculated_features[f'{name}-mad()'] = np.median(np.abs(sig - np.median(sig)))
        calculated_features[f'{name}-sma()'] = signal_magnitude_area(sig)
        calculated_features[f'{name}-energy()'] = signal_energy(sig)
        calculated_features[f'{name}-iqr()'] = iqr(sig)
        calculated_features[f'{name}-entropy()'] = signal_entropy(sig)
        for j, val in enumerate(arcoeff_proxy()):
            calculated_features[f'{name}-arCoeff(){j+1}'] = val

    fft_results = {}
    for name in ['tBodyAcc', 'tBodyAccJerk', 'tBodyGyro']:
        for i, axis in enumerate(['X', 'Y', 'Z']):
            fft_vals, freqs = compute_fft(signals_3d[name][:, i], FS_ACC)
            fft_results[f'f{name[1:]}-{axis}'] = (fft_vals, freqs)

            f_name = f'f{name[1:]}'
            calculated_features[f'{f_name}-mean()-{axis}'] = np.mean(fft_vals)
            calculated_features[f'{f_name}-std()-{axis}'] = np.std(fft_vals)
            calculated_features[f'{f_name}-meanFreq()-{axis}'] = weighted_mean_freq(fft_vals, freqs)
            calculated_features[f'{f_name}-energy()-{axis}'] = signal_energy(fft_vals)
            calculated_features[f'{f_name}-max()-{axis}'] = np.max(fft_vals)
            calculated_features[f'{f_name}-maxInds-{axis}'] = np.argmax(fft_vals)
            calculated_features[f'{f_name}-bandsEnergy()-1,16'] = bands_energy_proxy(fft_vals)
            calculated_features[f'{f_name}-bandsEnergy()-1,24'] = bands_energy_proxy(fft_vals)
            calculated_features[f'{f_name}-bandsEnergy()-9,16'] = bands_energy_proxy(fft_vals)

    gravity_mean = np.mean(gravity_acc, axis=0)
    body_acc_mean = np.mean(body_acc, axis=0)
    body_acc_jerk_mean = np.mean(body_acc_jerk, axis=0)

    calculated_features['angle(tBodyAccMean,gravity)'] = angle_proxy(body_acc_mean, gravity_mean)
    calculated_features['angle(tBodyAccJerkMean),gravityMean)'] = angle_proxy(body_acc_jerk_mean, gravity_mean)
    calculated_features['tBodyAcc-correlation()-X,Y'] = 0.0

    # Assemblage final
    final_vector = []
    feature_list_slim_92 = [     
        "tBodyAccJerk-std()-X", "tBodyAccJerkMag-energy()", "fBodyAccJerk-bandsEnergy()-1,16",
        "fBodyAccJerk-max()-X", "fBodyAccJerk-bandsEnergy()-1,24", "tBodyGyroJerk-mad()-Z",
        "fBodyAccJerk-bandsEnergy()-1,16", "fBodyAccJerk-std()-X", "fBodyAcc-entropy()-X",
        "fBodyAcc-mad()-X", "tBodyAccJerk-sma()", "fBodyAccJerk-mean()-X",
        "tBodyGyroJerk-sma()", "tBodyAccJerk-iqr()-X", "tBodyGyroJerk-iqr()-Z",
        "fBodyAccJerk-mean()-Y", "tBodyAcc-correlation()-X,Y", "tBodyAccJerkMag-sma()",
        "tBodyGyroJerk-iqr()-X", "tBodyAccJerk-mad()-Y", "fBodyAccJerk-max()-Y",
        "fBodyAcc-bandsEnergy()-9,16", "fBodyAccJerk-mad()-X", "fBodyAccJerk-bandsEnergy()-9,16",
        "tBodyAccJerk-entropy()-Y", "fBodyAccJerk-energy()-X", "tGravityAccMag-arCoeff()1",
        "tGravityAcc-arCoeff()-X,1", "fBodyAcc-energy()-X", "tBodyAccMag-arCoeff()1",
        "tGravityAcc-arCoeff()-X,3", "tBodyAcc-max()-X", "tGravityAcc-arCoeff()-X,2",
        "fBodyAcc-mean()-X", "fBodyAccMag-std()", "tBodyAccJerkMag-iqr()",
        "fBodyAccJerk-maxInds-X", "fBodyAcc-bandsEnergy()-1,16", "tGravityAcc-arCoeff()-X,4",
        "fBodyAccJerk-mad()-Y", "tBodyGyroJerkMag-mean()", "fBodyAccJerk-bandsEnergy()-1,8",
        "tBodyGyroJerk-iqr()-Y", "fBodyAccJerk-entropy()-Y", "tBodyAccJerkMag-mean()",
        "fBodyAccMag-entropy()", "tBodyAccJerk-entropy()-X", "fBodyAccJerk-entropy()-X",
        "tBodyAccJerk-energy()-X", "tBodyGyroJerk-entropy()-Z", "fBodyBodyAccJerkMag-mean()",
        "tBodyAccJerk-mad()-Z", "fBodyAcc-iqr()-X", "tBodyGyro-iqr()-Y",
        "tGravityAcc-arCoeff()-Z,1", "fBodyAcc-bandsEnergy()-1,8", "tGravityAcc-mad()-X",
        "tGravityAccMag-std()", "tGravityAcc-arCoeff()-Z,2", "tBodyAccMag-arCoeff()2",
        "fBodyAccMag-mad()", "fBodyAccMag-max()", "tBodyAccMag-mad()",
        "fBodyGyro-maxInds-Z", "tGravityAcc-mad()-Y", "fBodyAcc-bandsEnergy()-1,24",
        "fBodyAcc-std()-X", "tGravityAcc-arCoeff()-Y,1", "tGravityAcc-arCoeff()-Y,2",
        "fBodyGyro-maxInds-X", "tGravityAcc-arCoeff()-Y,4", "tGravityAcc-std()-X",
        "tGravityAcc-arCoeff()-Z,3", "tGravityAcc-entropy()-X", "tBodyAcc-std()-X",
        "fBodyAcc-max()-X", "tBodyAccMag-std()", "fBodyAccMag-meanFreq()",
        "tGravityAcc-arCoeff()-Y,3", "tGravityAccMag-mad()", "tGravityAccMag-sma()",
        "fBodyAccMag-energy()", "tBodyAccMag-energy()", "tBodyAccJerk-correlation()-X,Y",
        "tGravityAcc-std()-Y", "tBodyGyroJerk-mad()-X", "fBodyAccMag-sma()",
        "tGravityAcc-min()-X", "tGravityAcc-mean()-X", "tBodyAcc-mad()-X",
        "tGravityAcc-mad()-Z", "fBodyAcc-meanFreq()-Z"
    ]

    for name in feature_list_slim_92:
        final_vector.append(calculated_features.get(name, 0.0))

    return np.array(final_vector)

# --- SIMULATION / PLACEHOLDER DU MODELE SLIM (garde-le si tu as déjà un joblib) ---
def model_predict_nap(features):
    """
    Placeholder : si tu as un modèle joblib pour SLIM, charge-le une fois
    et fais : return int(slim_model.predict([features])[0])
    """
    if np.mean(np.abs(features)) < 0.5:
        return 0
    elif np.mean(np.abs(features)) < 1.5:
        return 1
    else:
        return 2

# --- DÉTECTEUR DE PRÉ-ALERTE (POC basé sur tendance) ---
def detect_pre_alert(fc_hist, spo2_hist, motion_hist, history_len=HISTORY_LEN):
    """
    Règle POC :
    - fenêtre glissante courte (history_len)
    - pré-alerte si :
      * SPO2 baisse >= 2% sur la fenêtre
      * ET/OU FC dérive durablement (> 8-10 bpm en fenêtre)
      * ET/OU diminution significative (>30%) de l'activité (motion mean) sur la fenêtre
    Retour : True/False
    """
    if len(fc_hist) < 3 or len(spo2_hist) < 3 or len(motion_hist) < 3:
        return False

    # utiliser la fenêtre la plus récente
    fc_win = np.array(fc_hist[-history_len:])
    spo2_win = np.array(spo2_hist[-history_len:])
    motion_win = np.array(motion_hist[-history_len:])

    # Tendances simples
    spo2_drop = spo2_win[0] - spo2_win[-1]   # positive si baisse
    fc_delta = fc_win[-1] - np.mean(fc_win[:max(1, len(fc_win)//2)])  # dernière vs première moitié
    motion_mean_start = np.mean(motion_win[:max(1, len(motion_win)//2)])
    motion_mean_end = np.mean(motion_win[max(1, len(motion_win)//2):])
    motion_drop_pct = (motion_mean_start - motion_mean_end) / (motion_mean_start + 1e-6)

    # Conditions
    cond_spo2 = spo2_drop >= 2.0 or spo2_win[-1] <= SEUIL_SPO2_LOW
    cond_fc = abs(fc_delta) >= 8.0  # dérive importante en bpm
    cond_motion = motion_drop_pct >= 0.30  # diminution d'activité >=30%

    # Log pour debugging
    logging.debug(f"PREALERT check: spo2_drop={spo2_drop:.2f}, fc_delta={fc_delta:.2f}, motion_drop_pct={motion_drop_pct:.2f}")

    # heuristique : si au moins deux conditions sont vraies -> pré-alerte
    score = sum([cond_spo2, cond_fc, cond_motion])
    return score >= 2

# --- ENDPOINT PRINCIPAL / pipeline complet ---
@app.post("/analyze_vitals_safe")
def analyze_vitals_safe(data: SensorData):
    try:
        # Normalisation sécurisée
        acc_buffer = np.tile(safe_normalize_acc([data.accel_x, data.accel_y, data.accel_z]), (400,1))
        gyro_buffer = np.tile(safe_normalize_gyro([data.gyro_x, data.gyro_y, data.gyro_z]), (400,1))

        # Estimation FC
        motion_magnitude = np.linalg.norm(acc_buffer, axis=1)
        max_motion = np.max(motion_magnitude)
        fc_corrigee = data.bpm * (1.1 if max_motion>2.5 else 1.05 if max_motion>1.5 else 1.0)
        fc_corrigee = 0.0 if not np.isfinite(fc_corrigee) else fc_corrigee

        # Features SLIM
        try:
            features_92 = calculate_slim_features(acc_buffer, gyro_buffer)
        except Exception as e:
            logging.error(f"Erreur SLIM: {e}")
            features_92 = np.zeros(92)

        # Prédiction mouvement
        try:
            nap_mouvement_pred = int(model_predict_nap(features_92))
        except Exception as e:
            logging.error(f"Erreur SLIM prediction: {e}")
            nap_mouvement_pred = 0

        nap_mouvement_pred = 0 if nap_mouvement_pred not in [0,1,2] else nap_mouvement_pred
        data.spo2 = SEUIL_SPO2_NORMAL if not np.isfinite(data.spo2) else data.spo2

        # Historique
        now_ts = data.timestamp.isoformat() if data.timestamp else datetime.now(timezone.utc).isoformat()
        fc_history.append(float(round(fc_corrigee,2)))
        spo2_history.append(float(round(data.spo2,2)))
        motion_history.append(nap_mouvement_pred)
        time_history.append(now_ts)

        # FIFO buffer
        if len(fc_history) > BUFFER_MAX_LEN:
            del fc_history[0:len(fc_history)-BUFFER_MAX_LEN]
            del spo2_history[0:len(spo2_history)-BUFFER_MAX_LEN]
            del motion_history[0:len(motion_history)-BUFFER_MAX_LEN]
            del time_history[0:len(time_history)-BUFFER_MAX_LEN]

        # Pré-alerte
        pre_alert = detect_pre_alert(fc_history, spo2_history, motion_history, history_len=min(HISTORY_LEN, len(fc_history)))

        # Classification
        statut_anomalie, alerte_active = classify_vitals_senior(nap_mouvement_pred, fc_corrigee, data.spo2, pre_alert)
        if statut_anomalie=="PreAlerte":
            alerte_active=False

        result_payload = {
            "timestamp": now_ts,
            "fc_corrigee_bpm": round(fc_corrigee,2),
            "nap_mouvement_code": int(nap_mouvement_pred),
            "statut_alerte": statut_anomalie,
            "is_critical": bool(alerte_active)
        }

        # Supabase insert
        if supabase:
            try:
                resp = supabase.table("vitals_analysis").insert(result_payload).execute()
                if getattr(resp, "data", None) is None:
                    logging.error(f"Supabase error: {getattr(resp,'error','unknown')}")
            except Exception as e:
                logging.error(f"Supabase insert error: {e}")

        return {"status":"OK","analysis":result_payload,"debug":{"pre_alert":pre_alert,"history_len":len(fc_history)}}

    except Exception as e:
        logging.error(f"Internal error analyze_vitals_safe: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")
