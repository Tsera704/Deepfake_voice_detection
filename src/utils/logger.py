import os
import logging

def get_project_logger(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    """
    Creates or retrieves a logger that routes to console + persistent log file.
    Colab: /content/drive/MyDrive/DeepFakeVoiceResearch/logs
    Local: ./logs (fallback when Drive is not mounted).
    """
    drive_log_dir = "/content/drive/MyDrive/DeepFakeVoiceResearch/logs"
    local_log_dir = os.path.join(os.getcwd(), "logs")
    log_dir = drive_log_dir if os.path.isdir("/content/drive") else local_log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
