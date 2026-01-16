import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter1d, median_filter
import librosa

def load_signal(filepath, target_sr=1000):
    # Loads .wav files and resamples it to target_sr (1000 Hz).
    try:
        # librosa handles loading and resampling
        audio, sr = librosa.load(filepath, sr=target_sr)
        return audio, sr
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

def schmidt_spike_removal(original_signal, fs):
    # 1. Define window (500ms)
    windowsize = int(round(fs / 2))
    
    # 2. Absolute signal
    abs_signal = np.abs(original_signal)
    
    # 3. Find the local peaks (Trajectory)
    traj = maximum_filter1d(abs_signal, size=windowsize)
    
    # 4. Use Median Filter to find the "average" background noise level
    # We compare the local peak (traj) to the median of the surrounding peaks
    background_level = median_filter(traj, size=windowsize)
    
    # 5. Define spikes: standard is > 3x the median background level
    # We add a small epsilon to ensure we don't flag floating-point noise in silent regions as spikes.
    spike_mask = traj > (3 * background_level + 0.0001)
    
    # 6. Remove spikes
    clean_signal = original_signal.copy()
    
    # Clamp the spikes to the max allowed value (3 * background)
    # We maintain the original sign (+/-) of the signal
    clean_signal[spike_mask] = np.sign(clean_signal[spike_mask]) * (3 * background_level[spike_mask])
    
    return clean_signal

def butterworth_bandpass(data, lowcut=25, highcut=450, fs=1000, order=4):
    # Applies a 4th order Butterworth bandpass filter (25-450 Hz).
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def normalize_signal(data):
    # Z-score normalization (Zero Mean, Unit Variance).
    return (data - np.mean(data)) / (np.std(data) + 1e-8)

def segment_signal(data, fs, window_size=5, overlap=0.5):
    # Cuts the audio into fixed 5-second windows.
    # If signal < 5s, it pads with zeros.
    # If signal > 5s, it creates overlapping chunks.
    window_samples = int(window_size * fs)
    step_samples = int(window_samples * (1 - overlap))
    
    segments = []
    
    # Case 1: Signal is shorter than window -> Pad it
    if len(data) < window_samples:
        padding = np.zeros(window_samples - len(data))
        padded_data = np.concatenate((data, padding))
        segments.append(padded_data)
    
    # Case 2: Signal is longer -> Slice it
    else:
        for start in range(0, len(data) - window_samples + 1, step_samples):
            end = start + window_samples
            segments.append(data[start:end])
            
    return np.array(segments)