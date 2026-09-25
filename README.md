# Urdu Speech - Stress & Lie Detection System

## System Components (6 Modules)

| # | Module | File | Library |
|---|--------|------|---------|
| 1 | Audio Input Module | `modules/audio_input.py` | Librosa |
| 2 | Speech to Text Module | `modules/speech_to_text.py` | Whisper |
| 3 | Feature Extraction Module | `modules/feature_extraction.py` | Librosa |
| 4 | Stress Detection Module | `modules/stress_detection.py` | ML (Random Forest/SVM/GB) |
| 5 | Lie Detection Module | `modules/lie_detection.py` | ML (feature-based scoring) |
| 6 | Evaluation & Reporting Module | `modules/evaluation_reporting.py` | Matplotlib + Pandas |

---

## Project Structure

```
urdu_speech_project/
├── backend/
│   ├── app.py                    # Main Flask API
│   ├── train_models.py           # ML Model training script
│   ├── modules/
│   │   ├── audio_input.py        # Module 1
│   │   ├── speech_to_text.py     # Module 2
│   │   ├── feature_extraction.py # Module 3
│   │   ├── stress_detection.py   # Module 4
│   │   ├── lie_detection.py      # Module 5
│   │   └── evaluation_reporting.py # Module 6
│   ├── models/
│   │   ├── stress_model.pkl      # Trained ML model
│   │   ├── label_encoder.pkl     # Label encoder
│   │   └── scaler.pkl            # Feature scaler
│   ├── dataset/                  # CSV dataset files
│   └── reports/                  # Generated charts/reports
├── frontend/
│   └── index.html                # Web UI
└── requirements.txt
```

---

## Setup & Run

### Step 1: Requirements Install Karein
```bash
pip install -r requirements.txt
```

### Step 2: Models Train Karein (ZAROOR chalayein)
```bash
cd urdu_speech_project/backend
python train_models.py
```

Yeh command:
- Dataset load karta hai (complete + males + females)
- Random Forest, Gradient Boosting, SVM train karta hai
- Best model select karta hai
- `models/` folder mein save karta hai
- `reports/` mein charts save karta hai

### Step 3: Server Chalayein
```bash
cd urdu_speech_project/backend
python app.py
```

### Step 4: Browser Mein Kholein
```
http://localhost:5000
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend (Web UI) |
| `/api/predict` | POST | Audio analyze karo |
| `/api/evaluate` | GET | Model evaluation run karo |
| `/test` | GET | API status check |

---

## ML Models Used (NO Deep Learning)

- **Random Forest Classifier** - n_estimators=300
- **Gradient Boosting Classifier** - n_estimators=200
- **Support Vector Machine (SVM)** - RBF kernel
- **Ensemble Voting Classifier** - Soft voting (best of all 3)

---

## Features Extracted (64 total)

- **MFCC (40)** - Voice texture/tone
- **Chroma (12)** - Pitch class energy
- **Spectral Contrast (7)** - Frequency peaks
- **ZCR (1)** - Zero Crossing Rate
- **Pitch / F0 (1)** - Fundamental frequency
- **Amplitude RMS (1)** - Voice energy
- **Duration (1)** - Audio length
- **Spectral Rolloff (1)** - High frequency content

---

## Reports Generated

After running `train_models.py` ya `/api/evaluate`:
- `reports/confusion_matrix.png`
- `reports/performance_metrics.png`
- `reports/model_comparison.png`
- `reports/emotion_distribution.png`
- `reports/model_metrics.csv`
- `reports/prediction_report_*.png` (har prediction ke baad)

---

## Note

- Pehle `train_models.py` chalana **zaroor** hai models ke liye
- Whisper model pehli baar internet se download hoga (~150MB)
- CPU pe chalane ke liye `fp16=False` already set hai
