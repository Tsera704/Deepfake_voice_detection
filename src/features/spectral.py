import numpy as np
import librosa

def extract_spectral(dsp_cache, sr, config):
    """Extracts raw spectral properties using the non-emphasized magnitude spectrogram."""
    S = dsp_cache['mag_raw']
    n_fft = dsp_cache['n_fft']
    hop_length = dsp_cache['hop_length']
    audio = dsp_cache['audio_raw']
    
    # Extract features directly from the shared spectrogram
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff_85 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    rolloff_95 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)
    flatness = librosa.feature.spectral_flatness(S=S)
    # Use frame_length matching the STFT parameters for RMS
    rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop_length)
    if zcr.shape[1] > S.shape[1]:
        zcr = zcr[:, :S.shape[1]]
    
    spectral_combined = np.vstack([
        centroid, bandwidth, rolloff_85, rolloff_95, flatness, rms, zcr
    ])
    return spectral_combined.T