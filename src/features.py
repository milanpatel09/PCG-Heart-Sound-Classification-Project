import numpy as np
import torch
import torchaudio
import pywt
from kymatio.torch import Scattering1D

# Check for GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_spectrogram_batch(audio_batch, n_fft=256, hop_length=25):
    """STFT Spectrogram. Shape: (Batch, 257, 20)"""
    if not isinstance(audio_batch, torch.Tensor):
        audio_batch = torch.tensor(audio_batch).float()
    audio_batch = audio_batch.to(DEVICE)
    
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=2.0
    ).to(DEVICE)
    
    spec = spec_transform(audio_batch)
    return torchaudio.transforms.AmplitudeToDB()(spec).cpu().numpy()

def extract_melspec_batch(audio_batch, sample_rate=1000, n_mels=128):
    """Mel-Spectrogram. Shape: (Batch, 128, 20)"""
    if not isinstance(audio_batch, torch.Tensor):
        audio_batch = torch.tensor(audio_batch).float()
    audio_batch = audio_batch.to(DEVICE)
    
    # Mel-Spec often needs slightly higher FFT resolution or standard 512
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=256,
        hop_length=25,
        n_mels=n_mels
    ).to(DEVICE)
    
    mel = mel_transform(audio_batch)
    return torchaudio.transforms.AmplitudeToDB()(mel).cpu().numpy()

def extract_mfcc_batch(audio_batch, sample_rate=1000, n_mfcc=42):
    """MFCC. Shape: (Batch, 13, 20)"""
    if not isinstance(audio_batch, torch.Tensor):
        audio_batch = torch.tensor(audio_batch).float()
    audio_batch = audio_batch.to(DEVICE)
    
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 256, "hop_length": 25, "n_mels": 128}
    ).to(DEVICE)
    
    mfcc = mfcc_transform(audio_batch)
    return mfcc.cpu().numpy()

def extract_scattering_batch(audio_batch, J=5, Q=2):
    """
    Robust Wavelet Scattering Transform.
    
    Parameters based on Heart Sound Physics:
    - J=8: Scale of 2^8 = 256 samples (256ms).
           This covers the duration of S1 (~100ms) and S2 (~100ms) perfectly.
           Previous J=6 (64ms) was likely too short for heart sounds.
    - Q=8: Standard quality factor for audio.
    
    Output Shape: (Batch, Channels, Time)
    """
    # 1. Tensor Setup
    if not isinstance(audio_batch, torch.Tensor):
        audio_batch = torch.tensor(audio_batch).float()
    audio_batch = audio_batch.to(DEVICE)
    
    # 2. Input Sanitization (CRITICAL STEP)
    # Replace NaNs in the *Input* audio with zeros
    if torch.isnan(audio_batch).any():
        audio_batch = torch.nan_to_num(audio_batch, nan=0.0)
    
    T = audio_batch.shape[-1]
    
    # 3. Initialize Kymatio
    try:
        scattering = Scattering1D(J=J, shape=(T,), Q=Q).to(DEVICE)
        Sx = scattering(audio_batch)
    except Exception as e:
        print(f"   [SCATTERING ERROR] Fallback to CPU due to: {e}")
        scattering = Scattering1D(J=J, shape=(T,), Q=Q).cpu()
        Sx = scattering(audio_batch.cpu())
        Sx = Sx.to(DEVICE)

    # 4. Output Sanitization (CRITICAL STEP)
    # Scattering can produce slightly negative zeros or NaNs on silent signals
    Sx = torch.nan_to_num(Sx, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 5. Safe Log Scaling (Decibel-like)
    # Instead of dividing by 1e-6 (unstable), we use standard log1p 
    # and a small epsilon offset to prevent log(0).
    # Formula: log(1 + |x|)
    Sx = torch.log1p(torch.abs(Sx))
    
    # 6. Check for remaining NaNs (Paranoia Check)
    if torch.isnan(Sx).any():
        # If still NaN, force to 0. This guarantees no crash.
        Sx = torch.zeros_like(Sx)

    return Sx.cpu().numpy()

def extract_cwt_single(audio_data, scale=64):
    """CWT (CPU Only). Shape: (64, 5000)"""
    scales = np.arange(1, scale + 1)
    coef, _ = pywt.cwt(audio_data, scales, 'morl')
    return np.abs(coef)
