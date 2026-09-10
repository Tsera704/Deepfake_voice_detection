import h5py
import numpy as np
from pathlib import Path


def load_cached_cqcc(h5_path, file_names, n_cqcc=20, max_frames=300):
    """
    Loads precomputed CQCC features from an HDF5 cache file on Google Drive.
    Returns a dictionary mapping file names to their corresponding CQCC numpy arrays.
    """
    cache_dict = {}

    if not Path(h5_path).exists():
        print(f"[WARNING] Cache file not found at: {h5_path}")
        return cache_dict

    with h5py.File(h5_path, 'r') as hf:
        if "cqcc" not in hf or "file_names" not in hf:
            print(f"[ERROR] Invalid cache structure in {h5_path}")
            return cache_dict

        stored_files = [f.decode('utf-8') if isinstance(f, bytes) else str(f) for f in hf["file_names"][:]]
        dset = hf["cqcc"]

        if not file_names:
            for i, fname in enumerate(stored_files):
                cache_dict[fname] = np.array(dset[i])
        else:
            file_to_idx = {fname: i for i, fname in enumerate(stored_files)}
            for fname in file_names:
                key = Path(fname).name
                if key in file_to_idx:
                    cache_dict[key] = np.array(dset[file_to_idx[key]])
                elif fname in file_to_idx:
                    cache_dict[fname] = np.array(dset[file_to_idx[fname]])

    return cache_dict
