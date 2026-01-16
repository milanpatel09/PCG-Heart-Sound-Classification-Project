import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


# Import your feature extractors
from src.features import (
    extract_spectrogram_batch, 
    extract_melspec_batch,
    extract_mfcc_batch, 
    extract_scattering_batch,
    extract_cwt_single
)

# --- CONFIGURATION ---
DATA_PATH = 'data/processed'
OUTPUT_DIR = 'visualizations'
GROUPS_PATH = os.path.join(DATA_PATH, 'groups.npy')
X_PATH = os.path.join(DATA_PATH, 'X_data.npy')
Y_PATH = os.path.join(DATA_PATH, 'y_data.npy')

# Feature List for loop
FEATURES = ['STFT', 'MelSpec', 'MFCC', 'Scattering', 'CWT']

def normalize(img):
    """Normalize for better contrast in plots"""
    return (img - img.min()) / (img.max() - img.min() + 1e-6)

def save_plot(feature_map, feature_name, recording_id, label, idx, save_folder):
    """Saves a single feature image without axes/borders (clean look)"""
    
    plt.figure(figsize=(10, 4))
    
    # Choose Colormap based on feature type
    cmap = 'inferno'
    if feature_name == 'MelSpec': cmap = 'magma'
    if feature_name == 'MFCC': cmap = 'viridis'
    if feature_name == 'CWT': cmap = 'jet'
    if feature_name == 'Scattering': cmap = 'plasma'

    # Plot
    plt.imshow(feature_map, origin='lower', aspect='auto', cmap=cmap)
    
    # Title info
    class_name = "Abnormal" if label == 1 else "Normal"
    plt.title(f"{feature_name} | ID: {recording_id} | {class_name}", fontsize=14, weight='bold')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    # Filename: Feature_Class_ID.png
    filename = f"{feature_name}_{class_name}_ID{recording_id}_idx{idx}.png"
    plt.savefig(os.path.join(save_folder, filename), dpi=150)
    plt.close()

def main():
    print(f"Loading data from {DATA_PATH}...")
    
    if not os.path.exists(X_PATH):
        raise FileNotFoundError("X_data.npy not found!")
        
    # Load data (Memory Map X to save RAM)
    X = np.load(X_PATH, mmap_mode='r')
    y = np.load(Y_PATH)
    groups = np.load(GROUPS_PATH)

    # --- SELECTION LOGIC ---
    np.random.seed(42) # Fixed seed = Same 10 patients every time
    
    normal_idxs = np.where(y == 0)[0]
    abnormal_idxs = np.where(y == 1)[0]
    
    # Select 5 unique random indices from each class
    sel_normal = np.random.choice(normal_idxs, 5, replace=False)
    sel_abnormal = np.random.choice(abnormal_idxs, 5, replace=False)
    
    # Combine them
    selection = np.concatenate([sel_normal, sel_abnormal])
    print(f"Selected 10 Indices: {selection}")
    
    # Create sub-folders for each feature
    for feat in FEATURES:
        os.makedirs(os.path.join(OUTPUT_DIR, feat), exist_ok=True)

    # --- PROCESSING LOOP ---
    print("\nGenering 50 Images...")
    
    for i, idx in enumerate(selection):
        # 1. Get Data
        raw_audio = X[idx]      # Shape (5000,)
        label = y[idx]
        rec_id = groups[idx]
        
        # Prepare Tensor batch (1, 5000) for the extractor functions
        audio_tensor = torch.tensor(raw_audio).float().unsqueeze(0)
        
        print(f"[{i+1}/10] Processing ID {rec_id} ({'Abnormal' if label==1 else 'Normal'})...")
        
        # 2. Extract All Features
        # Note: We take [0] because batch functions return (Batch, H, W)
        feat_stft = extract_spectrogram_batch(audio_tensor)[0]
        feat_mel  = extract_melspec_batch(audio_tensor)[0]
        feat_mfcc = extract_mfcc_batch(audio_tensor)[0]
        feat_scat = extract_scattering_batch(audio_tensor)[0]
        feat_cwt  = extract_cwt_single(raw_audio) # CWT takes numpy 1D
        
        # 3. Save Each Image to its respective folder
        save_plot(feat_stft, 'STFT',       rec_id, label, idx, os.path.join(OUTPUT_DIR, 'STFT'))
        save_plot(feat_mel,  'MelSpec',    rec_id, label, idx, os.path.join(OUTPUT_DIR, 'MelSpec'))
        save_plot(feat_mfcc, 'MFCC',       rec_id, label, idx, os.path.join(OUTPUT_DIR, 'MFCC'))
        save_plot(feat_scat, 'Scattering', rec_id, label, idx, os.path.join(OUTPUT_DIR, 'Scattering'))
        save_plot(feat_cwt,  'CWT',        rec_id, label, idx, os.path.join(OUTPUT_DIR, 'CWT'))

    print(f"\n✅ Done! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    # Ensure code uses CPU for plotting to avoid GPU conflict
    torch.device('cpu')
    main()