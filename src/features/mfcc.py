import numpy as np
import librosa

def extract_mfcc(dsp_cache, sr, config):
    """Extracts MFCCs using pre-computed emphasized Mel spectrogram."""
    
    # Bug #2 Fixed: Input is now a true log-power Mel spectrogram
    mfcc_static = librosa.feature.mfcc(
        S=librosa.power_to_db(dsp_cache['mel_emp']),
        sr=sr,
        n_mfcc=config['features']['mfcc']['n_mfcc']
    )
    
    # Compute Temporal Derivatives
    mfcc_delta = librosa.feature.delta(mfcc_static, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc_static, order=2)
    
    mfcc_combined = np.vstack([mfcc_static, mfcc_delta, mfcc_delta2])
    return mfcc_combined.T