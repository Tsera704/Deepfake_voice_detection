import numpy as np
import librosa
from scipy.fftpack import dct

def linear_filterbank(sr, n_fft, n_filters):
    """Constructs a linearly-spaced triangular filterbank."""
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    filter_freqs = np.linspace(0, sr / 2, n_filters + 2)
    
    weights = np.zeros((n_filters, int(1 + n_fft // 2)))
    for i in range(n_filters):
        left = filter_freqs[i]
        center = filter_freqs[i + 1]
        right = filter_freqs[i + 2]
        
        # Calculate the triangular up-slope and down-slope
        weights[i] = np.maximum(
            0, np.minimum((fft_freqs - left) / (center - left), 
                          (right - fft_freqs) / (right - center))
        )
    return weights

def extract_lfcc(dsp_cache, sr, config):
    """Extracts LFCCs using pre-computed emphasized power spectrogram."""
    n_lfcc = config['features']['lfcc']['n_lfcc']
    n_filters = config['features']['lfcc']['n_filters']
    
    # 1. Apply Linear Filterbank to the shared power spectrum
    lin_filters = linear_filterbank(sr, dsp_cache['n_fft'], n_filters)
    lin_spec = np.dot(lin_filters, dsp_cache['power_emp'])
    
    # 2. Logarithm (with safety guard) and DCT
    log_lin_spec = librosa.power_to_db(lin_spec, amin=1e-10)
    lfcc_static = dct(log_lin_spec, type=2, axis=0, norm='ortho')[:n_lfcc]
    
    # 3. Temporal Derivatives
    lfcc_delta = librosa.feature.delta(lfcc_static, order=1)
    lfcc_delta2 = librosa.feature.delta(lfcc_static, order=2)
    
    lfcc_combined = np.vstack([lfcc_static, lfcc_delta, lfcc_delta2])
    return lfcc_combined.T