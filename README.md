# PhysioGuard: Patient Deterioration Prediction System 🩺🚨

Welcome to **PhysioGuard**, an advanced AI pipeline designed to predict patient deterioration (like sepsis, shock, or cardiorespiratory failure) *before* it happens.

This repository holds a battle-tested, state-of-the-art machine learning framework that combines tabular clinical data (like age, gender, admission type) with temporal time-series vital signs (like heart rate, blood pressure, etc.) to evaluate patient risk in real-time.

---

## 🌟 What makes this special?

This isn't just a standard machine learning model. This pipeline includes several advanced modules natively integrated to make it robust for real-world hospital deployment:

1. **Self-Supervised Learning (SSL):** The deep learning model first learns the "language of vitals" by predicting masked (hidden) vital signs before it ever sees the deterioration labels. This helps it understand human physiology.
2. **Federated Learning (FL):** The model simulates training across multiple independent hospitals (silos) without sharing their raw patient data, using the `FedAvg` algorithm.
3. **Differential Privacy (DP):** Adds statistical noise to the deep learning gradients (via Opacus) so that the AI cannot accidentally memorize specific patient records.
4. **CatBoost + TCN-Transformer Ensemble:** Fuses an incredibly powerful Gradient Boosted Tree (CatBoost) with a Temporal Convolutional Transformer to get the best of both tabular and sequential data.
5. **Domain Generalization (LODO):** Tests the model's ability to perform on entirely unseen hospital units (Leave-One-Domain-Out validation) to guarantee it isn't just memorizing specific hospital artifacts.
6. **Explainability (XAI):** Uses SHAP values and Captum to explain *why* it made a prediction (e.g., "Lactate is rising rapidly").

---

## 📂 Project Structure

```text
Patient-Deterioration-Prediction-System/
│
├── configs/                  # Configuration files that dictate how the AI trains
│   ├── default.yaml          # The main, heavy-duty 2-day training configuration
│   └── quick_test.yaml       # A tiny configuration for testing if the code works in 5 mins
│
├── core/                     # Fundamental data and evaluation code
│   ├── data_loader.py        # Reads the CSV and creates 250+ advanced clinical features
│   ├── features.py           # Feature engineering logic (rolling averages, SOFA scores, etc.)
│   ├── logger.py             # Creates the detailed logs you see in the terminal
│   └── metrics.py            # Calculates PR-AUC, ROC-AUC, Calibration, etc.
│
├── dataset/                  # Put your raw data here!
│   └── train.csv             # The raw patient data (Required)
│
├── models/                   # The AI Brains
│   ├── catboost_model.py     # Gradient Boosting model
│   ├── ensemble.py           # Blends CatBoost and Deep Learning together
│   ├── lstm_attention.py     # Alternative RNN-based deep learning model
│   └── tcn_transformer.py    # The main Deep Learning sequence model
│
├── modules/                  # Advanced Real-World Hospital Features
│   ├── calibration/          # Adjusts confidence scores so 90% actually means 90%
│   ├── cross_validation/     # Tests the model on different chunks of data
│   ├── deployment/           # Packages the model for ONNX/OpenVINO
│   ├── differential_privacy/ # Opacus integration for patient privacy
│   ├── domain_generalization/# Leave-One-Domain-Out testing
│   ├── fairness/             # Ensures the model isn't biased against age or gender
│   ├── federated_learning/   # Simulates training across multiple separated hospital servers
│   ├── ssl/                  # Self-Supervised pre-training (Masked prediction)
│   └── xai/                  # Generates Explainable AI graphs (SHAP)
│
├── pipelines/                # The main scripts to run everything
│   └── run_full_pipeline.py  # Run this to start the magic!
│
└── artifacts/                # Generated folder where all models, logs, and graphs are saved!
```

---

## 🚀 How to Run the System

### 1. Requirements
* **Python 3.10** is strictly required.
* **Hardware:** Any modern CPU will work, but for the main training run, a GPU (NVIDIA CUDA or AMD DirectML) with at least 12GB VRAM is highly recommended. (Note: Differential privacy DP-SGD doesn't run on DirectML, but the pipeline will automatically detect this and skip it safely while keeping everything else hardware-accelerated).

### 2. Testing the Waters (Smoke Test)
Before dedicating days to training, do a quick run on 5% of the data to make sure your environment is perfect:
```powershell
py -3.10 pipelines/run_full_pipeline.py --config configs/quick_test.yaml
```
If this finishes successfully and creates a folder in `artifacts/`, you are ready!

### 3. The Grand Championship Run
To start the massive, highly reliable training cycle (which may take several days depending on hardware):
```powershell
py -3.10 pipelines/run_full_pipeline.py --config configs/default.yaml
```
*Tip: Leave your computer plugged in, prevent it from sleeping, and check the terminal logs (`pipeline.log`) to track its progress!*

---

## ⚙️ Configuration Guide (`default.yaml`)

The `default.yaml` file is the master control panel. We have tuned it for maximum clinical reliability, but here is what the settings mean if you want to play with them:

* `max_rows`: Set to `null` to use all 293,000+ patient records. Set to `1000` to just test locally.
* `device: "auto"`: Automatically decides whether to use CUDA, DirectML (AMD), or CPU.
* `features.use_advanced: true`: Enables the generation of 250+ extra features (like 12-hour rolling averages and vital sign acceleration).
* `modules.deep_learning.epochs`: Controls how long the Transformer neural network trains. We default to `150` for deep understanding.
* `modules.federated_learning.clients`: Simulates the number of hospitals (e.g., `5`).

## 📊 Where do the results go?

After a successful run, look inside the `artifacts/run_[TIMESTAMP]/` directory. You will find:
1. `pipeline.log`: Every single step and score documented.
2. `ssl_pretrained_tcntransformer.pt`: The neural network's initial understanding of human biology.
3. `dl_checkpoint_latest.pt`: The final neural network weights.
4. `model/`: Contains the `ensemble.pkl` which is the final fused AI.
5. `shap_summary.png`: A beautiful graph showing exactly which clinical features the AI thinks are most responsible for patient deterioration.
6. `lodo_results.csv`: Mathematical proof of how well the model works on unseen hospital units.

Enjoy saving lives! 🩺
