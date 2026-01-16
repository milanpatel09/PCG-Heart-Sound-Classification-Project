import numpy as np
import torch
import torchaudio
import pywt
from kymatio.torch import Scattering1D

# Check for GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_spectrogram_batch(audio_batch, n_fft=256, hop_length=25):
    """STFT Spectrogram. Shape: (Batch, 129, 201)"""
    if not isinstance(audio_batch, torch.Tensor):
        audio_batch = torch.tensor(audio_batch).float()
    audio_batch = audio_batch.to(DEVICE)
    
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=2.0
    ).to(DEVICE)
    
    spec = spec_transform(audio_batch)
    return torchaudio.transforms.AmplitudeToDB()(spec).cpu().numpy()

def extract_melspec_batch(audio_batch, sample_rate=1000, n_mels=128):
    """Mel-Spectrogram. Shape: (Batch, 128, 201)"""
    if not isinstance(audio_batch, torch.Tensor):
        audio_batch = torch.tensor(audio_batch).float()
    audio_batch = audio_batch.to(DEVICE)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=256,
        hop_length=25,
        n_mels=n_mels
    ).to(DEVICE)
    
    mel = mel_transform(audio_batch)
    return torchaudio.transforms.AmplitudeToDB()(mel).cpu().numpy()


#n_mfcc=42 chosen to cover a wider range of cepstral coefficients for heart sounds
def extract_mfcc_batch(audio_batch, sample_rate=1000, n_mfcc=13):
    """MFCC. Shape: (Batch, 13, 201)"""
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


# --- 1. OPTIMIZED CWT (Physics-Informed) ---
def extract_cwt_single(audio_data, sample_rate=1000, f_min=20, f_max=500, n_scales=64):
    """
    CWT with Log-spaced scales tuned to Heart Sound Physics (20-500Hz).
    
    Why this is better:
    - Current: Linear scales 1-64 covers ~12Hz to 800Hz unevenly.
    - New: Log-spacing ensures equal attention to low freq (S1/S2) and high freq (Murmurs).
    """
    # Morlet wavelet center frequency
    w = pywt.ContinuousWavelet('morl')
    
    # SAFETY FIX: If pywt returns None, use the standard constant 0.8125
    if w.center_frequency is None:
        center_freq = 0.8125 
    else:
        center_freq = w.center_frequency

    # Calculate scales for exactly 20Hz to 500Hz
    # Scale = (Center_Freq * Sample_Rate) / Frequency
    min_scale = (center_freq * sample_rate) / f_max
    max_scale = (center_freq * sample_rate) / f_min
    
    # Logarithmic spacing (powers of 2 style) is better for audio than linear
    scales = np.geomspace(min_scale, max_scale, num=n_scales)
    
    coef, _ = pywt.cwt(audio_data, scales, 'morl')
    
    # Output shape: (64, 5000)
    return np.abs(coef)

# --- 2. OPTIMIZED SCATTERING (Event-Based) ---
def extract_scattering_batch(audio_batch, sample_rate=1000, J=7, Q=2):
    """
    Scattering Transform tuned to Heart Event Duration.
    
    Parameters Explained:
    - J=7 (128 samples @ 1000Hz = 128ms):
      * S1/S2 sounds last ~100-120ms.
      * J=7 captures the full "shape" of the heart sound in one coefficient.
      * Q=2 (Filters per octave):
    """
    if not isinstance(audio_batch, torch.Tensor):
        audio_batch = torch.tensor(audio_batch).float()
    audio_batch = audio_batch.to(DEVICE)
    
    # Input Sanitization
    if torch.isnan(audio_batch).any():
        audio_batch = torch.nan_to_num(audio_batch, nan=0.0)
    
    T = audio_batch.shape[-1]
    
    try:
        # Initialize Kymatio with updated J
        scattering = Scattering1D(J=J, shape=(T,), Q=Q).to(DEVICE)
        Sx = scattering(audio_batch)
    except Exception as e:
        print(f"   [SCATTERING ERROR] Fallback to CPU: {e}")
        scattering = Scattering1D(J=J, shape=(T,), Q=Q).cpu()
        Sx = scattering(audio_batch.cpu())
        Sx = Sx.to(DEVICE)

    # Output Sanitization
    Sx = torch.nan_to_num(Sx, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Log1p scaling (Safe dB)
    Sx = torch.log1p(torch.abs(Sx))

    # 6. Check for remaining NaNs (Paranoia Check)
    if torch.isnan(Sx).any():
        # If still NaN, force to 0. This guarantees no crash.
        Sx = torch.zeros_like(Sx)
    
    return Sx.cpu().numpy()