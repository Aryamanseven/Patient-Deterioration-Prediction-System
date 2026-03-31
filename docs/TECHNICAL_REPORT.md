# Technical Report: PhysioGuard Remote Patient Monitoring

## 1. Problem Context & Objective
Dependent elderly individuals and neurological patients (e.g., stroke, dementia) are at high risk of silent physiological decline. Subtle early warning signs—changes in heart rate, respiratory rate, blood pressure, etc.—are often missed in home-care settings until emergency hospitalization is required.

Our objective is to predict patient physiological deterioration within a 6 to 12-hour window using historical, continuous time-series vital signs. We aim to deploy this as a remote patient monitoring (RPM) dashboard, leveraging edge-capable, privacy-preserving machine learning.

---

## 2. Model Architecture & Core Innovations

The system relies on a dual-pathway ensemble, combining classical gradient boosting with a sprawling, generalized Deep Learning pipeline.

### A. Deep Learning (DL): TCN-Transformer Hybrid
*   **What we get from it:** While standard models look at a "snapshot" of a patient's vitals, the Temporal Convolutional Network (TCN) captures immediate, high-frequency spikes (e.g., a sudden jump in heart rate). The Transformer layer acts as a long-term memory mechanism, recognizing slow, multiday deterioration trends.
*   **The Benefit:** We capture the *chronological trajectory* of the patient's health, rather than just isolated numbers.

### B. Self-Supervised Learning (SSL)
*   **What we get from it:** SSL forces the Transformer to "mask" certain hours of a patient's vitals and guess what those missing vitals are, purely to learn the underlying rhythm of a human body, *before* we ever tell it who crashed and who survived.
*   **The Benefit:** It massively boosts accuracy on small datasets because the model already mathematically understands human physiology before the actual training begins.

### C. Federated Learning (FL)
*   **What we get from it:** FL simulates training the AI directly on isolated "silos" (e.g., patients' smartwatches or separate home-care networks). Only the cryptographic mathematical weights are merged on the central cloud server via `FedAvg`.
*   **The Benefit:** Absolute Data Privacy. We never centralize or expose the elderly patient's highly sensitive, raw biological and home data.

### D. Differential Privacy (DP)
*   **What we get from it:** Provided by Meta's `Opacus`, DP injects mathematical Gaussian noise into the gradients during the localized Federated Learning process.
*   **The Benefit:** It acts as an encryption lock. Even if a cyber-attacker intercepts the mathematical weights transmitted from an elderly patient's home, it is mathematically impossible to reverse-engineer their specific identity or routine.

---

## 3. Evaluation Metrics Explained

In clinical machine learning, standard accuracy is dangerously misleading (if 99% of patients survive, a broken AI that *always* predicts survival is technically 99% accurate, but kills the 1% who crash). We rely on stringent medical metrics:

### 1. PR-AUC (Precision-Recall Area Under Curve)
*   **What it measures:** How well the model handles heavily imbalanced data (many healthy hours vs. a few crash hours). It balances Precision (how many of our alarms were real?) and Recall (how many real crashes did we successfully catch?).
*   **Target:** **HIGHER is better.** (0.0 to 1.0). In an ICU setting with 5% deterioration rates, anything above **0.45** is considered exceptional, state-of-the-art predictive performance.

### 2. ROC-AUC (Receiver Operating Characteristic)
*   **What it measures:** The model's general ability to separate the two classes (Healthy vs. Deteriorating) across all possible alarm thresholds.
*   **Target:** **HIGHER is better.** (0.5 is a random coin flip, 1.0 is perfect). A clinical-grade model should comfortably sit above **0.80**.

### 3. Brier Score (Calibration)
*   **What it measures:** The mathematical confidence of the model. If the dashboard tells a nurse the patient has a "90% risk of crashing", does that patient actually crash exactly 9 times out of 10?
*   **Target:** **LOWER is better.** (0.0 is perfect, 1.0 is totally wrong). A Brier score below **0.10** is elite, meaning the AI is highly trustworthy and not overconfident.

### 4. Sensitivity / Recall
*   **What it measures:** Out of 100 actual real-world patient crashes, how many did the AI successfully flag in advance?
*   **Target:** **HIGHER is better.** Missing a crash is fatal. We want Sensitivity as high as possible, ideally above **0.85**, even if it means slightly more false alarms.

---

## 4. System Limitations & Future Work

While standard algorithms assume high-fidelity continuous tracking, real-world wearable sensors are prone to disconnection and battery death. 

*   **Future Innovation 1 (GAN Sensor-Fusion):** Implementing a Generative Adversarial Network (GAN) to flawlessly impute and "hallucinate" missing multi-hour blocks of data when an elderly patient removes their smartwatch to shower or charge the device.
*   **Future Innovation 2 (Edge Compilation):** Compiling our PyTorch TCN-Transformer into `ONNX` or `TensorFlow Lite` to run the inference cycle locally on a low-power microcontroller or Apple Watch, preventing system failure during a home WiFi outage.
