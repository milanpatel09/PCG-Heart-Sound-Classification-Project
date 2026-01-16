import os
import numpy as np
import torch
from tqdm import tqdm
from src.features import (
    extract_spectrogram_batch, 
    extract_melspec_batch,
    extract_mfcc_batch, 
    extract_scattering_batch,
    extract_cwt_single
)

# CONFIGURATION
PROCESSED_DATA_PATH = 'data/processed'
FEATURES_PATH = 'data/features'
BATCH_SIZE = 100 

def create_memmap(folder, filename, shape, dtype=np.float32):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)

def run_stage2():
    X_path = os.path.join(PROCESSED_DATA_PATH, 'X_data.npy')
    if not os.path.exists(X_path):
        print("Error: X_data.npy not found.")
        return

    X = np.load(X_path, mmap_mode='r')
    N_SAMPLES = X.shape[0]
    print(f"Loaded X_data: {X.shape}")
    print(f"Using Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # 1. DRY RUN FOR SHAPES 
    print("\nPerforming Dry Run to determine Feature Shapes...")
    dummy_input = torch.zeros(1, 5000) # 1 sample, 5000 timepoints
    dummy_np = np.zeros(5000)
    
    # Get actual shapes dynamically
    shape_spec = extract_spectrogram_batch(dummy_input).shape[1:]
    shape_mel = extract_melspec_batch(dummy_input).shape[1:]
    shape_mfcc = extract_mfcc_batch(dummy_input).shape[1:]
    shape_scat = extract_scattering_batch(dummy_input).shape[1:]
    shape_cwt = extract_cwt_single(dummy_np).shape 

    print(f"STFT Shape: {shape_spec}")
    print(f"Mel-Spec Shape: {shape_mel}")
    print(f"MFCC Shape: {shape_mfcc}")
    print(f"Scattering Shape: {shape_scat}")
    print(f"CWT Shape: {shape_cwt}")

    # 2. INITIALIZE MEMMAPS
    print("\nInitializing Master Matrices on SSD...")
    fp_spec = create_memmap(os.path.join(FEATURES_PATH, 'spectrogram'), 'spectrogram.npy', (N_SAMPLES, *shape_spec))
    fp_mel = create_memmap(os.path.join(FEATURES_PATH, 'melspec'), 'melspec.npy', (N_SAMPLES, *shape_mel))
    fp_mfcc = create_memmap(os.path.join(FEATURES_PATH, 'mfcc'), 'mfcc.npy', (N_SAMPLES, *shape_mfcc))
    fp_scat = create_memmap(os.path.join(FEATURES_PATH, 'scattering'), 'scattering.npy', (N_SAMPLES, *shape_scat))
    
    print(f"Allocating CWT (~{N_SAMPLES * 64 * 5000 * 4 / 1e9:.2f} GB)...")
    fp_cwt = create_memmap(os.path.join(FEATURES_PATH, 'cwt'), 'cwt.npy', (N_SAMPLES, *shape_cwt))

    # 3. BATCH PROCESSING
    print(f"\nStarting Extraction (Batch Size: {BATCH_SIZE})...")
    num_batches = int(np.ceil(N_SAMPLES / BATCH_SIZE))
    
    for b in tqdm(range(num_batches)):
        start_idx = b * BATCH_SIZE
        end_idx = min((b + 1) * BATCH_SIZE, N_SAMPLES)
        
        # Load batch to RAM
        batch_signals = np.array(X[start_idx:end_idx]) 
        
        # GPU Features
        fp_spec[start_idx:end_idx] = extract_spectrogram_batch(batch_signals)
        fp_mel[start_idx:end_idx] = extract_melspec_batch(batch_signals)
        fp_mfcc[start_idx:end_idx] = extract_mfcc_batch(batch_signals)
        fp_scat[start_idx:end_idx] = extract_scattering_batch(batch_signals)
        
        # CPU Feature (CWT)
        cwt_list = [extract_cwt_single(sig) for sig in batch_signals]
        fp_cwt[start_idx:end_idx] = np.array(cwt_list)
        
        if b % 10 == 0:
            fp_spec.flush(); fp_mel.flush(); fp_mfcc.flush(); fp_scat.flush(); fp_cwt.flush()

    # Final Flush
    fp_spec.flush(); fp_mel.flush(); fp_mfcc.flush(); fp_scat.flush(); fp_cwt.flush()
    print("\nStage 2 Complete. All 5 features saved.")

if __name__ == "__main__":
    run_stage2()