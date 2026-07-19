import h5py
import numpy as np
import os
import json

def init_hdf5(file_path, feature_dim):
    """
    Initializes an expandable HDF5 file layout structurally isolated by split profiles.
    
    Args:
        file_path (str): Local or Drive path to target file.
        feature_dim (int): Total unified feature size (~3270).
    """
    if os.path.exists(file_path):
        print(f"[INFO] File {file_path} already exists. Skipping initialization.")
        return
        
    with h5py.File(file_path, 'w') as hf:
        # Create clear structural attributes
        hf.attrs['feature_dimension'] = feature_dim
        print(f"[INITIALIZED] Empty HDF5 container created at {file_path} with structural dim {feature_dim}")

def append_chunk(file_path, group_name, features, labels, file_ids, checkpoint_path):
    """
    Appends processed feature groups to disk dynamically. 
    Maintains infinite layout growth configurations via unbounded maxshape definitions.
    """
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    encoded_ids = [fid.encode('utf-8') for fid in file_ids]
    
    with h5py.File(file_path, 'a') as hf:
        if group_name not in hf:
            grp = hf.create_group(group_name)
            grp.create_dataset('features', data=features, maxshape=(None, features.shape[1]), chunks=True)
            grp.create_dataset('labels', data=labels, maxshape=(None,), chunks=True)
            str_type = h5py.string_dtype(encoding='utf-8')
            grp.create_dataset('file_ids', data=encoded_ids, maxshape=(None,), dtype=str_type, chunks=True)
        else:
            grp = hf[group_name]
            curr_size = grp['features'].shape[0]
            addition_size = features.shape[0]
            new_size = curr_size + addition_size
            
            # Unfold target datasets dynamically
            grp['features'].resize((new_size, features.shape[1]))
            grp['labels'].resize((new_size,))
            grp['file_ids'].resize((new_size,))
            
            # Write chunk contents straight to disk
            grp['features'][curr_size:new_size] = features
            grp['labels'][curr_size:new_size] = labels
            grp['file_ids'][curr_size:new_size] = encoded_ids
            
        # Explicit flush to guarantee data survives kernel disconnects
        hf.flush()
            
    print(f"[WRITE SUCCESS] Appended {features.shape[0]} items to {group_name}. New dataset size: {new_size}")

def read_hdf5_split(file_path, group_name):
    """Extracts datasets into clean, accessible NumPy matrices."""
    with h5py.File(file_path, 'r') as hf:
        if group_name not in hf:
            raise KeyError(f"Data split group '{group_name}' was not found inside {file_path}")
        
        features = hf[group_name]['features'][:]
        labels = hf[group_name]['labels'][:]
        file_ids = [fid.decode('utf-8') for fid in hf[group_name]['file_ids'][:]]
        
    return features, labels, file_ids