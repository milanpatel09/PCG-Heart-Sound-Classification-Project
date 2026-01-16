# PCG Heart Sound Classification (M.Tech Thesis Project)

This repository contains the complete implementation of an end-to-end deep learning pipeline for classifying Phonocardiogram (PCG) heart sound signals into Normal and Abnormal classes using time–frequency representations and a ResNet-18 model.

The project focuses on comparing multiple time–frequency features under a strict, leakage-free evaluation protocol.

---

## Dataset

This project uses the **PhysioNet 2016 Heart Sound Dataset**.

Download it from:
[https://physionet.org/content/challenge-2016/1.0.0/](https://physionet.org/content/challenge-2016/1.0.0/)

### Folder Setup

After downloading, copy the following folders from the dataset:

* training-a
* training-b
* training-c
* training-d
* training-e
* training-f

Then place them like this inside your cloned repository:

```
PCG-Heart-Sound-Classification-Thesis/
│
├── data/
│   └── raw/
│       ├── training-a/
│       ├── training-b/
│       ├── training-c/
│       ├── training-d/
│       ├── training-e/
│       └── training-f/
```

---

## Features Compared

Five time–frequency representations are evaluated:

* STFT Spectrogram
* Mel-Spectrogram
* MFCC
* Continuous Wavelet Transform (CWT)
* Wavelet Scattering Transform (WST)

All features are converted to image-like inputs and classified using ResNet-18.

---

## Environment Setup

### 1. Clone the repository

```
git clone https://github.com/milanpatel09/PCG-Heart-Sound-Classification-Thesis.git
cd PCG-Heart-Sound-Classification-Thesis
```

### 2. Create virtual environment

```
python -m venv env
```

Activate:

* Windows:

```
.\env\Scripts\activate
```

* Linux/Mac:

```
source env/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Running the Pipeline (Stage-wise)

The project is divided into three main stages:

---

### Stage 1: Preprocessing

This stage:

* Resamples audio from 2000 Hz → 1000 Hz
* Applies Schmidt spike removal
* Applies Butterworth band-pass filter (25–400 Hz)
* Performs Z-score normalization
* Segments audio into 5s clips with 2.5s overlap
* Saves output as NumPy arrays

Run:

```
python stage1_main.py
```

Output:

```
data/processed/X_data.npy   # shape: (24450, 5000)
data/processed/Y_data.npy
```

---

### Stage 2: Feature Extraction

This stage converts segments into five time–frequency features:

* STFT Spectrogram
* Mel-Spectrogram
* MFCC
* CWT
* Wavelet Scattering Transform

Run:

```
python stage2_main.py
```

Output files:

```
data/features/spectrogram.npy
data/features/mel-spec.npy
data/features/mfcc.npy
data/features/cwt.npy
data/features/scattering.npy
```

---

### Stage 3: Classification (No Leakage Setting)

This stage:

* Uses ResNet-18
* Converts features to 3-channel images
* Applies stratified, recording-wise split (80/20)
* Trains model and selects best epoch

Run:

```
python stage3_classification.py
```

Models saved in:

```
models_checkpoints/
```

---

### Optional: Leakage Experiment

To see optimism bias when leakage is allowed:

```
python stage3_leaky.py
```

This performs segmentation before splitting, allowing data leakage.

---

## Evaluation Metrics

Models are evaluated using:

* Validation Accuracy
* Sensitivity
* Specificity
* F1-Score
* Mean Accuracy = (Sensitivity + Specificity) / 2

Best epoch is selected using priority:

```
M.Acc > Sensitivity > F1 > Specificity > Val.Acc
```

---

## Key Findings

* Best feature for 2D CNN (ResNet-18):
  Mel-Spectrogram > MFCC > CWT > STFT > WST

* Data leakage inflates performance by 5–10%.

* Feature–model compatibility is more important than model depth.

---

## Future Work

* Feature-level fusion using CNN confidence scores (logits)
* Hybrid ML classifier on combined CNN outputs
* Cross-dataset validation
* Explainable AI integration (Grad-CAM, LIME)

---

## Citation

If you use this work, please cite:

Milan Arvind Patel (2026)
"Time-Frequency Analysis of Heart Sound Signals: A Comparative Study Using Convolutional Neural Networks"
M.Tech Thesis, IIIT Bhubaneswar
