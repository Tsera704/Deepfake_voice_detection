import numpy as np
import librosa
from scipy.stats import skew, kurtosis

def compute_shared_dsp(audio, sr, config):
    """Computes foundational DSP artifacts once to prevent redundant STFT calls."""
    n_fft = int((config['pipeline']['frame_length_ms'] / 1000.0) * sr)
    hop_length = int((config['pipeline']['frame_stride_ms'] / 1000.0) * sr)
    window = config['pipeline']['window_type']
    
    # 1. Raw STFT (For true Spectral features)
    stft_raw = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
    mag_raw = np.abs(stft_raw)
    
    # 2. Emphasized Signal & Linear Power STFT (For LFCC)
    audio_emp = librosa.effects.preemphasis(audio, coef=config['pipeline']['pre_emphasis'])
    stft_emp = librosa.stft(audio_emp, n_fft=n_fft, hop_length=hop_length, window=window)
    power_emp = np.abs(stft_emp) ** 2
    
    # 3. Mel Spectrogram (For MFCC) - Built from the emphasized power spectrum
    mel_emp = librosa.feature.melspectrogram(
        S=power_emp, 
        sr=sr, 
        n_mels=config['features']['mfcc']['n_mels']
    )
    
    return {
        'audio_raw': audio,
        'mag_raw': mag_raw,
        'power_emp': power_emp,
        'mel_emp': mel_emp, 
        'n_fft': n_fft,
        'hop_length': hop_length
    }

def compute_utterance_stats(feature_matrix, stats_list):
    """
    Collapses a (frames x features) matrix into a structured 1D array of moments.
    Uses NaN-safe operations to survive zero-padded digital silence blocks.
    
    Args:
        feature_matrix (np.ndarray): 2D matrix from an extractor module.
        stats_list (list): Statistical directives (e.g., ['mean', 'std', 'skew', 'p95']).
        
    Returns:
        np.ndarray: 1D aggregated array.
    """
    calculated_stats = []
    
    for stat in stats_list:
        if stat == 'mean':
            calculated_stats.append(np.nanmean(feature_matrix, axis=0))
        elif stat == 'std':
            calculated_stats.append(np.nanstd(feature_matrix, axis=0))
        elif stat == 'skew':
            # omit ignores NaN slices rather than corrupting the entire file matrix
            calculated_stats.append(skew(feature_matrix, axis=0, nan_policy='omit'))
        elif stat == 'kurtosis':
            calculated_stats.append(kurtosis(feature_matrix, axis=0, nan_policy='omit'))
        elif stat == 'min':
            calculated_stats.append(np.nanmin(feature_matrix, axis=0))
        elif stat == 'max':
            calculated_stats.append(np.nanmax(feature_matrix, axis=0))
        elif stat == 'median':
            calculated_stats.append(np.nanmedian(feature_matrix, axis=0))
        elif stat.startswith('p'):
            # Parse custom percentiles (e.g., 'p95' -> 95.0) on the fly
            percentile_val = float(stat.replace('p', ''))
            calculated_stats.append(np.nanpercentile(feature_matrix, percentile_val, axis=0))
            
    return np.concatenate(calculated_stats)

def fuse_all_features(audio, sr, config, extractors):
    """Orchestrates pipeline using a shared DSP cache."""
    fused_vector = []
    
    # Generate the shared computations once
    dsp_cache = compute_shared_dsp(audio, sr, config)
    
    for feat_name, extract_fn in extractors.items():
        # Pass the cache instead of raw audio
        feat_matrix = extract_fn(dsp_cache, sr, config)
        
        # Bug #1 Fixed: Pulling from the global 'stats' list
        feat_stats = compute_utterance_stats(feat_matrix, config['stats'])
        fused_vector.append(feat_stats)
        
    return np.concatenate(fused_vector)