import os
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold


def parse_protocol_speaker_groups(protocol_path, audio_dir, file_ext=".flac"):
    """
    Groups file paths by speaker_id from ASVspoof2019 LA protocol.
    
    Protocol format: speaker_id file_id system_id eval_set label
    Returns: dict mapping speaker_id -> list of {path, label, file_id}
    """
    speaker_files = defaultdict(list)
    
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                speaker_id = parts[0]
                file_id = parts[1]
                label = parts[4]  # bonafide or spoof
                
                # Construct audio path (matching notebook pattern)
                audio_path = os.path.join(audio_dir, f"{file_id}{file_ext}")
                
                speaker_files[speaker_id].append({
                    'path': audio_path,
                    'label': 0 if label == 'bonafide' else 1,
                    'file_id': file_id
                })
    
    return dict(speaker_files)


def speaker_stratified_kfold(speaker_files, n_splits=5, random_state=42):
    """
    Creates speaker-independent K-fold splits.
    
    Each fold: train = all speakers EXCEPT fold_idx validation/test speakers.
    Ensures no speaker appears in both train and validation/test sets.
    
    Returns list of fold dicts with train/test file lists and speaker IDs.
    """
    speakers = list(speaker_files.keys())
    labels = []

    for spk in speakers:
        files = speaker_files[spk]
        spk_labels = [e['label'] for e in files]
        majority = int(round(float(np.mean(spk_labels)))) if spk_labels else 0
        labels.append(majority)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_speaker_idx, test_speaker_idx) in enumerate(skf.split(speakers, labels)):
        train_speakers = [speakers[i] for i in train_speaker_idx]
        test_speakers = [speakers[i] for i in test_speaker_idx]
        
        # Verify no speaker leakage
        assert not set(train_speakers) & set(test_speakers), "Speaker leakage detected!"
        
        # Flatten to file lists
        train_files = []
        for spk in train_speakers:
            train_files.extend(speaker_files[spk])
        
        test_files = []
        for spk in test_speakers:
            test_files.extend(speaker_files[spk])
        
        folds.append({
            'fold': fold_idx,
            'train_files': train_files,
            'train_labels': [e['label'] for e in train_files],
            'test_files': test_files,
            'test_labels': [e['label'] for e in test_files],
            'train_speakers': train_speakers,
            'test_speakers': test_speakers,
            'n_train_samples': len(train_files),
            'n_test_samples': len(test_files),
            'n_train_speakers': len(train_speakers),
            'n_test_speakers': len(test_speakers)
        })
    
    return folds


def create_train_val_test_split(speaker_files, 
                                n_splits=5, 
                                val_fold_idx=0,
                                random_state=42):
    """
    Creates train/val/test split using first fold as validation,
    remaining folds combined as train, or can use nested split.
    
    Common pattern: 
    - Outer 5-fold CV for final evaluation
    - Inner split: 1 fold as val, 4 folds as train
    """
    folds = speaker_stratified_kfold(speaker_files, n_splits=n_splits, random_state=random_state)
    
    # Use first fold as validation, remaining as training
    val_fold = folds[val_fold_idx]
    train_folds = [folds[i] for i in range(n_splits) if i != val_fold_idx]
    
    # Combine all non-val folds as training data
    train_files = []
    for fold in train_folds:
        train_files.extend(fold['train_files'])
    
    # Combine train fold labels
    train_labels = []
    for fold in train_folds:
        train_labels.extend(fold['train_labels'])
    
    # Use test fold as held-out test set
    test_files = val_fold['test_files']
    test_labels = val_fold['test_labels']
    
    return {
        'train_files': train_files,
        'train_labels': train_labels,
        'val_files': val_fold['test_files'],
        'val_labels': val_fold['test_labels'],
        'test_files': test_files,
        'test_labels': test_labels,
        'train_speakers': list(set([s for f in train_files for s in [f.split("/")[-1].split(".")[0]]])),
        'val_speakers': list(set([s for f in val_fold['test_files'] for s in [f.split("/")[-1].split(".")[0]]])),
        'test_speakers': list(set([s for f in test_files for s in [f.split("/")[-1].split(".")[0]]])),
        'n_train': len(train_files),
        'n_val': len(val_fold['test_files']),
        'n_test': len(test_files)
    }