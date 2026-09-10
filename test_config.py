import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'features.yaml')) as f:
    config = yaml.safe_load(f)

SMOKE_MODE = config.get('smoke_test_mode', False)
print(f'Smoke test mode: {SMOKE_MODE}')

print(f'Stats: {config.get("stats")}')
print(f'Sample rate: {config.get("sample_rate")}')

cqcc = config['features']['cqcc']
print(f'CQCC use_cached: {cqcc["use_cached"]}')

st = config['smoke_test']
print(f'Smoke n_bonafide: {st["n_bonafide"]}')
print(f'Smoke n_spoof: {st["n_spoof"]}')
print(f'Smoke max_files: {st["max_files"]}')
print(f'Smoke feat_dim: {st.get("feat_dim")}')

from src.utils.cache_loader import load_cached_cqcc
print('cache_loader OK')
from src.features.fusion import compute_shared_dsp, compute_utterance_stats, fuse_features_for_file
print('fusion OK')
from src.utils.hdf5_io import init_hdf5, append_chunk, read_hdf5_split
print('hdf5_io OK')

print('\nConfig OK!')
