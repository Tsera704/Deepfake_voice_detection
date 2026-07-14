import os
import json
from google.colab import drive

def mount_drive():
    """Mounts Google Drive to the default Colab path."""
    print("[INFO] Mounting Google Drive...")
    drive.mount('/content/drive')

def verify_paths(base_path="/content/drive/MyDrive/DeepFakeVoiceResearch"):
    """Validates and builds target output directories on the Shared Drive."""
    required_dirs = [
        "datasets/ASVspoof2019_LA",
        "datasets/WaveFake",
        "datasets/In_The_Wild",
        "features",
        "models/mlruns",
        "results",
        "figures"
    ]
    print(f"[INFO] Verifying directory tree at: {base_path}")
    for folder in required_dirs:
        full_path = os.path.join(base_path, folder)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            print(f"[CREATED] {full_path}")
    print("[INFO] Core directories verified.")

def get_kaggle_credentials(secret_json_path="/content/drive/MyDrive/kaggle.json"):
    """Sets up environment variables required by the Kaggle API wrapper."""
    if not os.path.exists(secret_json_path):
        raise FileNotFoundError(f"Missing kaggle.json at {secret_json_path}. Please place it there.")
    
    with open(secret_json_path, 'r') as f:
        creds = json.load(f)
        
    os.environ['KAGGLE_USERNAME'] = creds['username']
    os.environ['KAGGLE_KEY'] = creds['key']
    print("[INFO] Kaggle API environmental variables set up successfully.")