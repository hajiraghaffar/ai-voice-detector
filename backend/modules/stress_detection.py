"""
=====================================
MODULE 4: Stress Detection Module
=====================================
Classifies stress level from acoustic features.
Uses: Machine Learning ONLY (Random Forest + SVM + Gradient Boosting + XGBoost)
Levels: Low, Medium, High
"""

import numpy as np
import joblib
import os

# Stress mapping from emotion labels
EMOTION_TO_STRESS = {
    "angry":   "High",
    "fear":    "High",
    "disgust": "High",
    "sad":     "Medium",
    "neutral": "Low",
    "happy":   "Low",
}

# Stress numeric scores for calculations
STRESS_SCORES = {
    "Low":    1,
    "Medium": 2,
    "High":   3
}

STRESS_COLORS = {
    "Low":    "#27ae60",   # green
    "Medium": "#f39c12",   # orange
    "High":   "#e74c3c",   # red
}


def engineer_features_for_prediction(X):
    """
    Apply the SAME advanced feature engineering used in training.
    This is CRITICAL - features must match what the model was trained on!
    
    CSV Feature Layout (64 features):
    [0-3]   Duration & basic stats
    [4-7]   RMS Energy stats
    [8-11]  ZCR stats
    [12-15] Spectral Centroid stats
    [16-19] Spectral Rolloff stats
    [20-23] Spectral Bandwidth stats
    [24-62] MFCCs 13x3 (39 features)
    [63]    Chroma mean
    
    Args:
        X: shape (1, 64) or (n_samples, 64)
    """
    X_eng = X.copy()
    n_samples = X.shape[0]
    
    # ========== MFCC Statistics (Emotional Spectrum) ==========
    # MFCCs are at indices 24-62 (13 coeffs x 3 stats = 39 features)
    mfccs = X[:, 24:63]
    X_eng = np.column_stack([
        X_eng,
        np.mean(mfccs, axis=1),        # Mean MFCC
        np.std(mfccs, axis=1),         # Std MFCC
        np.max(mfccs, axis=1),         # Max MFCC
        np.min(mfccs, axis=1),         # Min MFCC
        np.median(mfccs, axis=1),      # Median MFCC
        np.percentile(mfccs, 25, axis=1),    # Q1
        np.percentile(mfccs, 75, axis=1),    # Q3
    ])
    
    # ========== Pitch & Energy Features (Critical for Happy vs Sad) ==========
    if X.shape[1] >= 64:
        # Spectral centroid mean (index 12) as pitch proxy
        pitch_proxy = X[:, 12]
        # RMS mean (index 4) as energy
        energy = X[:, 4]
        # ZCR mean (index 8)
        zcr = X[:, 8]
        
        # Pitch variation indicator
        pitch_variation = np.zeros(n_samples)
        for i in range(n_samples):
            pitch_variation[i] = np.std([pitch_proxy[i]]) if pitch_proxy[i] != 0 else 0
        
        X_eng = np.column_stack([
            X_eng,
            pitch_proxy,                       # Pitch proxy (spectral centroid)
            pitch_variation,                   # Pitch variation indicator
            energy,                            # Voice energy (RMS)
            np.abs(energy * pitch_proxy),      # Energy-pitch interaction
            zcr,                               # Zero crossing rate
            pitch_proxy / (energy + 1e-6),     # Pitch-to-energy ratio
        ])
    
    # ========== Spectral Features (Happy = More Brightness) ==========
    # Upper MFCCs: indices 51-62 (MFCC coeff 9-12, last 12 features of MFCC block)
    upper_mfcc = X[:, 51:63]
    X_eng = np.column_stack([
        X_eng,
        np.mean(upper_mfcc, axis=1),        # High-frequency content
        np.std(upper_mfcc, axis=1),         # Spectral variability
    ])
    
    # ========== Temporal Dynamics ==========
    if n_samples > 1:
        mfcc_delta = np.gradient(mfccs, axis=0)  # Rate of change along samples
        X_eng = np.column_stack([
            X_eng,
            np.mean(np.abs(mfcc_delta), axis=1),  # Temporal activity
            np.std(mfcc_delta, axis=1),             # Temporal variability
        ])
    else:
        # Single sample: no temporal variation, use zeros
        X_eng = np.column_stack([
            X_eng,
            np.zeros(n_samples),  # Temporal activity
            np.zeros(n_samples),  # Temporal variability
        ])
    
    return X_eng


def load_stress_model(model_dir: str = "models"):
    """
    Load trained ML model, scaler and label encoder.
    
    Returns:
        tuple: (model, label_encoder, scaler)
    """
    model_path = os.path.join(model_dir, "stress_model.pkl")
    le_path    = os.path.join(model_dir, "label_encoder.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    for path in [model_path, le_path, scaler_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Please run 'python train_models.py' first!"
            )

    model  = joblib.load(model_path)
    le     = joblib.load(le_path)
    scaler = joblib.load(scaler_path)

    print(f"[Stress Detection] ML Model loaded: {type(model).__name__}")
    return model, le, scaler


def predict_stress(features: np.ndarray, model, le, scaler) -> dict:
    """
    Detect stress using ML model.
    
    Args:
        features: np.ndarray shape (1, 64) - RAW features from extraction
        model: Trained ML classifier
        le: LabelEncoder
        scaler: StandardScaler
    
    Returns:
        dict: emotion, stress_level, stress_score, confidence, probabilities
    """
    # IMPORTANT: Apply feature engineering BEFORE scaling!
    features_engineered = engineer_features_for_prediction(features)

    # Validate dimensions match scaler
    n_expected = scaler.n_features_in_
    n_actual = features_engineered.shape[1]
    if n_actual != n_expected:
        raise ValueError(
            f"Feature dimension mismatch! "
            f"Model expects {n_expected} features, got {n_actual}. "
            f"Input features shape: {features.shape}, "
            f"Engineered shape: {features_engineered.shape}"
        )

    # Scale features
    features_scaled = scaler.transform(features_engineered)

    # Predict emotion
    pred_encoded = model.predict(features_scaled)[0]
    emotion = le.inverse_transform([pred_encoded])[0]

    # Get probabilities (confidence)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(proba))
        all_classes = le.classes_
        prob_dict = {cls: round(float(p), 3) for cls, p in zip(all_classes, proba)}
    else:
        confidence = 1.0
        prob_dict = {emotion: 1.0}

    # Map emotion to stress level
    stress_level = EMOTION_TO_STRESS.get(emotion.lower(), "Medium")
    stress_score = STRESS_SCORES[stress_level]

    print(f"[Stress Detection]")
    print(f"  Emotion Detected : {emotion}")
    print(f"  Stress Level     : {stress_level}")
    print(f"  ML Confidence    : {confidence:.2%}")
    print(f"  Probabilities    : {prob_dict}")

    return {
        "emotion": emotion,
        "stress_level": stress_level,
        "stress_score": stress_score,
        "stress_color": STRESS_COLORS[stress_level],
        "confidence": round(confidence, 3),
        "emotion_probabilities": prob_dict
    }


def get_stress_description(stress_level: str) -> str:
    """Stress level explanation in English."""
    descriptions = {
        "Low": (
            "No stress detected in your voice. "
            "Calm and relaxed state detected."
        ),
        "Medium": (
            "Some stress detected. "
            "Mild tension or sad emotion present."
        ),
        "High": (
            "High stress detected. "
            "Strong negative emotion (anger/fear) detected."
        )
    }
    return descriptions.get(stress_level, "Unknown stress level")
