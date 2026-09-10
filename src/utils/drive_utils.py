import os
import json


def mount_drive():
    """Mounts Google Drive on Colab; no-op locally."""
    try:
        from google.colab import drive
    except ImportError:
        print("[INFO] Not on Colab (google.colab unavailable), skipping mount.")
        return
    print("[INFO] Mounting Google Drive...")
    drive.mount('/content/drive')


def verify_paths(base_path="/content/drive/MyDrive/DeepFakeVoiceResearch"):
    """Validates and builds target output directories."""
    if not os.path.isdir("/content/drive") and base_path.startswith("/content/drive"):
        base_path = os.getcwd()
        print(f"[INFO] Drive not mounted, using local base: {base_path}")
    required_dirs = [
        "datasets/smoke_test/flac",
        "datasets/ASVspoof2019_LA",
        "datasets/In_The_Wild",
        "features",
        "models/mlruns",
        "results",
        "figures",
        "logs",
    ]
    print(f"[INFO] Verifying directory tree at: {base_path}")
    for folder in required_dirs:
        full_path = os.path.join(base_path, folder)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            print(f"[CREATED] {full_path}")
    print("[INFO] Core directories verified.")
    return base_path


def get_kaggle_credentials(secret_json_path="/content/drive/MyDrive/kaggle.json"):
    """Sets up environment variables required by the Kaggle API wrapper."""
    candidates = [
        secret_json_path,
        os.path.join(os.getcwd(), "configs", "kaggle.json"),
        os.path.expanduser("~/.kaggle/kaggle.json"),
    ]
    resolved = next((p for p in candidates if os.path.exists(p)), None)
    if resolved is None:
        raise FileNotFoundError(
            f"Missing kaggle.json. Checked: {candidates}. Place it in configs/kaggle.json."
        )

    with open(resolved, 'r') as f:
        creds = json.load(f)

    os.environ['KAGGLE_USERNAME'] = creds['username']
    os.environ['KAGGLE_KEY'] = creds['key']
    print(f"[INFO] Kaggle API variables set from {resolved}.")
