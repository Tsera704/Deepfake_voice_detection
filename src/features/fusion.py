import numpy as np
import librosa
from scipy.stats import skew, kurtosis
from src.utils.logger import get_project_logger
from src.utils.hdf5_io import init_hdf5, append_chunk, read_hdf5_split
import os

logger = get_project_logger("FeatureFusion", log_file="extraction.log")

__all__ = [
    "load_cached_cqcc", "init_hdf5", "append_chunk", "read_hdf5_split",
    "compute_shared_dsp", "compute_utterance_stats",
    "fuse_all_features", "fuse_features_for_file",
]


def load_cached_cqcc(cache_path_or_dir: str, file_list: list, n_cqcc: int = 20, max_frames: int = 300) -> dict:
    if os.path.isdir(cache_path_or_dir):
        cache_path = os.path.join(cache_path_or_dir, "cqcc_cache_train.h5")
    else:
        cache_path = cache_path_or_dir

    if not os.path.exists(cache_path):
        logger.warning(f"CQCC cache not found at {cache_path}, skipping CQCC fusion")
        return {}

    from src.utils.cache_loader import load_cached_cqcc as ll
    cqcc_cache = ll(cache_path, file_list, n_cqcc=n_cqcc, max_frames=max_frames)

    for f in list(cqcc_cache.keys()):
        feat = np.asarray(cqcc_cache[f])
        if feat.shape[0] < max_frames:
            pad_width = max_frames - feat.shape[0]
            feat = np.pad(feat, ((0, pad_width), (0, 0)), mode='constant')
        else:
            feat = feat[:max_frames, :]
        cqcc_cache[f] = feat.astype(np.float32)

    return cqcc_cache


def compute_shared_dsp(audio, sr, config):
    """Computes foundational DSP artifacts once to prevent redundant STFT calls."""
    n_fft = int((config['pipeline']['frame_length_ms'] / 1000.0) * sr)
    hop_length = int((config['pipeline']['frame_stride_ms'] / 1000.0) * sr)
    window = config['pipeline']['window_type']

    stft_raw = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
    mag_raw = np.abs(stft_raw)

    audio_emp = librosa.effects.preemphasis(audio, coef=config['pipeline']['pre_emphasis'])
    stft_emp = librosa.stft(audio_emp, n_fft=n_fft, hop_length=hop_length, window=window)
    power_emp = np.abs(stft_emp) ** 2

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
    """
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    calculated_stats = []

    for stat in stats_list:
        if stat == 'mean':
            calculated_stats.append(np.nanmean(feature_matrix, axis=0))
        elif stat == 'std':
            calculated_stats.append(np.nanstd(feature_matrix, axis=0))
        elif stat == 'skew':
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
            percentile_val = float(stat.replace('p', ''))
            calculated_stats.append(np.nanpercentile(feature_matrix, percentile_val, axis=0))

    fused = np.concatenate(calculated_stats)
    return np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _resolve_stats(config):
    stats_def = config.get('stats', ['mean', 'std'])
    if isinstance(stats_def, dict):
        return stats_def.get('functions', ['mean', 'std'])
    return stats_def


def fuse_all_features(audio, sr, config, extractors, cqcc_mat=None):
    """Orchestrates pipeline using a shared DSP cache + optional per-file CQCC matrix."""
    logger.info("---Starting feature fusion---")
    stats_list = _resolve_stats(config)
    dsp_cache = compute_shared_dsp(audio, sr, config)
    fused_vector = []

    for feat_name, extract_fn in extractors.items():
        feat_matrix = extract_fn(dsp_cache, sr, config)
        feat_stats = compute_utterance_stats(feat_matrix, stats_list)
        fused_vector.append(feat_stats)

    use_cached = config.get('features', {}).get('cqcc', {}).get('use_cached', False)
    if use_cached and cqcc_mat is not None:
        fused_vector.append(compute_utterance_stats(np.asarray(cqcc_mat), stats_list))

    return np.concatenate(fused_vector)


def fuse_features_for_file(audio, sr, config, extractors, cqcc_mat=None):
    """Fuses features for a single file. Pass that file's CQCC matrix, not the whole cache dict."""
    stats_list = _resolve_stats(config)
    dsp_cache = compute_shared_dsp(audio, sr, config)

    frame_level_features = []
    for feat_name, extract_fn in extractors.items():
        feat_matrix = np.asarray(extract_fn(dsp_cache, sr, config))
        if feat_matrix.ndim == 2 and feat_matrix.shape[0] < feat_matrix.shape[1]:
            pass
        frame_level_features.append(feat_matrix)

    if frame_level_features:
        min_frames = min(f.shape[0] for f in frame_level_features)
        trimmed = [f[:min_frames, :] for f in frame_level_features]
        combined = np.hstack(trimmed)
    else:
        combined = np.zeros((0, 0), dtype=np.float32)

    if cqcc_mat is not None:
        max_frames = config.get('features', {}).get('cqcc', {}).get('max_frames', 300)
        cqcc_feat = np.asarray(cqcc_mat)
        if cqcc_feat.shape[0] < max_frames:
            cqcc_feat = np.pad(cqcc_feat, ((0, max_frames - cqcc_feat.shape[0]), (0, 0)), mode='constant')
        else:
            cqcc_feat = cqcc_feat[:max_frames, :]
        stats_cqcc = compute_utterance_stats(cqcc_feat, stats_list)
        stats_live = compute_utterance_stats(combined, stats_list) if combined.size else np.array([], dtype=np.float32)
        return np.concatenate([stats_live, stats_cqcc] if stats_live.size else [stats_cqcc])

    return compute_utterance_stats(combined, stats_list)
