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
import os # Import pour les variables d'environnement
from datetime import datetime


# Configuration logging
logging.basicConfig(level=logging.INFO)

# --- PARAMÈTRES ET CLÉS DE BASE DE DONNÉES ---
# Lecture des variables d'environnement (Render)
SUPABASE_URL = os.getenv("SUPABASE_URL", "URL_PAR_DEFAUT_SI_MANQUANTE")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "CLE_PAR_DEFAUT_SI_MANQUANTE")
SEUIL_FC_STRESS = float(os.getenv("SEUIL_FC_STRESS", 90.0))

# Paramètres du contrat IoT/Cloud
FS_BVP = 100.0
FS_ACC = 50.0
WINDOW_SEC = 8
ORDER = 3
MIN_HR_HZ = 40 / 60
MAX_HR_HZ = 220 / 60

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

# Modèle de données attendu de l'ESP32
class SensorData(BaseModel):
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    bpm: float
    spo2: float
    timestamp: datetime 

app = FastAPI()

# --- FONCTIONS DE BASE DE FEATURE ENGINEERING (Module 1) ---

def butter_lowpass_filter(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return filtfilt(b, a, data, axis=0)

def compute_jerk(signal, fs):
    jerk = np.diff(signal, axis=0, prepend=signal[0:1]) * fs
    return jerk

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

def signal_entropy(signal):
    hist, _ = np.histogram(signal, bins=10, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist)) if len(hist) > 1 else 0.0

def weighted_mean_freq(fft_vals, freqs):
    return np.sum(fft_vals * freqs) / np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0

def bands_energy_proxy(fft_vals):
    return np.sum(fft_vals**2) / np.max(fft_vals) if np.max(fft_vals) > 0 else 0.0

def angle_proxy(v1, v2):
    norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_prod == 0: return 0.0
    return np.dot(v1, v2) / norm_prod

def arcoeff_proxy():
    # Coefficients d'Autoregression (PROXY pour la démo)
    return [0.1, 0.05, 0.02, 0.01]

# --- FONCTION DE CORRECTION CARDIAQUE (Module 2) ---

def estimate_fc_corrected(bvp_signal, acc_data):
    num_bvp_samples = bvp_signal.size
    window_samples = int(WINDOW_SEC * FS_BVP)
    step_samples = int(WINDOW_SEC * FS_BVP / 4)

    acc_resampled = np.zeros((num_bvp_samples, 3))
    try:
        # Rééchantillonnage de l'ACC (50Hz) à la fréquence du BVP (100Hz)
        for i in range(3):
            acc_resampled[:, i] = resample(acc_data[:, i], num_bvp_samples)
    except IndexError:
        return 0.0
    except ValueError:
        return 0.0

    motion_magnitude = np.sqrt(np.sum(acc_resampled**2, axis=1))
    hr_estimates = []

    # Traitement par fenêtres glissantes
    for start in range(0, num_bvp_samples - window_samples, step_samples):
        end = start + window_samples

        bvp_window = bvp_signal[start:end]
        acc_window = motion_magnitude[start:end]

        if len(bvp_window) != window_samples: continue

        # Calcul du Spectrogramme (PSD) pour BVP et Mouvement
        f_bvp, _, Pxx_den_bvp = spectrogram(bvp_window, FS_BVP, nperseg=window_samples, noverlap=0, mode='psd')
        f_acc, _, Pxx_den_acc = spectrogram(acc_window, FS_BVP, nperseg=window_samples, noverlap=0, mode='psd')

        Pxx_bvp = Pxx_den_bvp.flatten(); Pxx_acc = Pxx_den_acc.flatten()

        # Identification des fréquences dans la plage FC (40-220 BPM)
        acc_hr_indices = np.where((f_acc >= MIN_HR_HZ) & (f_acc <= MAX_HR_HZ))[0]
        motion_peaks = f_acc[acc_hr_indices][np.argsort(Pxx_acc[acc_hr_indices])[-2:]]
        bvp_hr_indices = np.where((f_bvp >= MIN_HR_HZ) & (f_bvp <= MAX_HR_HZ))[0]
        Pxx_bvp_hr = Pxx_den_bvp[bvp_hr_indices]; f_bvp_hr = f_bvp[bvp_hr_indices]

        if len(f_bvp_hr) == 0: continue

        candidate_hr_freq = None
        max_power = -1

        # LOGIQUE DE CORRECTION: Éliminer les pics BVP coïncidant avec le mouvement
        for i, freq in enumerate(f_bvp_hr):
            # Si la fréquence BVP est trop proche d'un pic de mouvement (0.1 Hz)
            is_motion = any(np.abs(freq - mp) < 0.1 for mp in motion_peaks)

            if not is_motion and Pxx_bvp_hr[i] > max_power:
                max_power = Pxx_bvp_hr[i]
                candidate_hr_freq = freq

        # Fallback: Si aucune fréquence non-motion n'est trouvée, prendre la plus puissante
        if candidate_hr_freq is None:
             candidate_hr_freq = f_bvp_hr[np.argmax(Pxx_bvp_hr)]

        hr_estimates.append(candidate_hr_freq * 60)

    if not hr_estimates:
        return 0.0
    return np.mean(hr_estimates)


# --- FONCTION DE CALCUL DES 92 FEATURES (Module 1) ---
# NOTE: Cette fonction utilise des PROXYs pour les features complexes (arCoeff, bandsEnergy, etc.)
def calculate_slim_features(acc_raw_400, gyro_raw_400):

    # Séparation Gravité / Corps (Filtre Passe-Bas à 0.3 Hz)
    gravity_acc = butter_lowpass_filter(acc_raw_400, 0.3, FS_ACC, ORDER)
    body_acc = acc_raw_400 - gravity_acc

    # Jerk (Dérivée)
    body_acc_jerk = compute_jerk(body_acc, FS_ACC)
    body_gyro_jerk = compute_jerk(gyro_raw_400, FS_ACC)

    # Magnitudes (1D)
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

    # 1. CALCUL STANDARD (Time Domain & Magnitude)
    for name, sig in signals_3d.items():
        for i, axis in enumerate(['X', 'Y', 'Z']):
            calculated_features[f'{name}-std()-{axis}'] = np.std(sig[:, i])
            calculated_features[f'{name}-mad()-{axis}'] = np.median(np.abs(sig[:, i] - np.median(sig[:, i])))
            calculated_features[f'{name}-entropy()-{axis}'] = signal_entropy(sig[:, i])
            calculated_features[f'{name}-iqr()-{axis}'] = iqr(sig[:, i])
            calculated_features[f'{name}-energy()-{axis}'] = signal_energy(sig[:, i])
            calculated_features[f'{name}-mean()-{axis}'] = np.mean(sig[:, i])
            calculated_features[f'{name}-max()-{axis}'] = np.max(sig[:, i])
            for j, val in enumerate(arcoeff_proxy()): # arCoeff PROXY
                 calculated_features[f'{name}-arCoeff()-{axis},{j+1}'] = val

    for name, sig in signals_1d.items():
        calculated_features[f'{name}-mean()'] = np.mean(sig)
        calculated_features[f'{name}-std()'] = np.std(sig)
        calculated_features[f'{name}-mad()'] = np.median(np.abs(sig - np.median(sig)))
        calculated_features[f'{name}-sma()'] = signal_magnitude_area(sig)
        calculated_features[f'{name}-energy()'] = signal_energy(sig)
        calculated_features[f'{name}-iqr()'] = iqr(sig)
        calculated_features[f'{name}-entropy()'] = signal_entropy(sig)
        for j, val in enumerate(arcoeff_proxy()): # arCoeff PROXY
             calculated_features[f'{name}-arCoeff(){j+1}'] = val

    # 2. CALCUL FFT DOMAIN
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

            # PROXY pour bandsEnergy
            calculated_features[f'{f_name}-bandsEnergy()-1,16'] = bands_energy_proxy(fft_vals)
            calculated_features[f'{f_name}-bandsEnergy()-1,24'] = bands_energy_proxy(fft_vals)
            calculated_features[f'{f_name}-bandsEnergy()-9,16'] = bands_energy_proxy(fft_vals)

    # 3. CALCUL DES ANGLES & CORRELATION
    gravity_mean = np.mean(gravity_acc, axis=0)
    body_acc_mean = np.mean(body_acc, axis=0)
    body_acc_jerk_mean = np.mean(body_acc_jerk, axis=0)

    calculated_features['angle(tBodyAccMean,gravity)'] = angle_proxy(body_acc_mean, gravity_mean)
    calculated_features['angle(tBodyAccJerkMean),gravityMean)'] = angle_proxy(body_acc_jerk_mean, gravity_mean)
    calculated_features['tBodyAcc-correlation()-X,Y'] = 0.0 # PROXY

    # 4. ASSEMBLAGE DU VECTEUR FINAL DE 92 FEATURES (DANS L'ORDRE STRICT)
    final_vector = []
    # Liste réelle des 92 features (copiée du document fourni précédemment)
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
        # Tente d'ajouter la feature calculée. Si elle n'est pas présente (à cause d'un PROXY manquant), ajoute 0.0.
        final_vector.append(calculated_features.get(name, 0.0))

    return np.array(final_vector)


# --- FONCTION DE PRÉDICTION NAP (Module 1 - Simulation du Modèle) ---

def model_predict_nap(features):
    # Simuler la logique du Random Forest SLIM entraîné.
    # EN PRODUCTION: Charger le modèle via joblib.load('modele_slim.joblib')
    if np.mean(np.abs(features)) < 0.5:
         return 0 # Repos
    elif np.mean(np.abs(features)) < 1.5:
         return 1 # Léger
    else:
         return 2 # Intense


# --- CONTRÔLEUR PRINCIPAL (Cloud Function) ---

@app.post("/analyze_vitals")
def analyze_vitals(data: SensorData):
    try:
        # Génération des buffers pour SLIM (simulation)
        ppg_buffer = np.array([data.bpm] * 800)  # remplacer 800 par la taille attendue
        acc_buffer = np.array([[data.accel_x, data.accel_y, data.accel_z]] * 400)
        gyro_buffer = np.array([[data.gyro_x, data.gyro_y, data.gyro_z]] * 400)

        # Estimation FC corrigée
        fc_corrigee = estimate_fc_corrected(ppg_buffer, acc_buffer)

        # Calcul des features et prédiction NAP
        features_92 = calculate_slim_features(acc_buffer, gyro_buffer)
        nap_mouvement = model_predict_nap(features_92)

        # Classification finale
        statut_anomalie = "Normal"
        alerte_active = False
        if nap_mouvement == 0 and fc_corrigee > SEUIL_FC_STRESS:
            statut_anomalie = "ALERTE: Stress Mental Sévère"
            alerte_active = True

        # Payload Supabase
        result_payload = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "fc_corrigee_bpm": round(fc_corrigee, 2),
            "nap_mouvement_code": int(nap_mouvement),
            "statut_alerte": statut_anomalie,
            "is_critical": alerte_active
        }

        if supabase:
            response = supabase.table('vitals_analysis').insert(result_payload).execute()
            if response.data is None:
                logging.error(f"Erreur Supabase: {response.error}")

        return {"status": "OK", "analysis": result_payload}

    except Exception as e:
        logging.error(f"Erreur d'exécution IA: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne de traitement IA: {str(e)}")
