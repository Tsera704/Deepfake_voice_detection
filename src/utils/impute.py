import numpy as np
import logging
from sklearn.impute import SimpleImputer

# Setup isolated corruption logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataPipelineImputation")

class SafeImputer:
    def __init__(self, strategy="most_frequent"):
        # Enforcing valid scikit-learn parameter to prevent pipeline crashes
        self.strategy = "most_frequent" 
        self.imputer = SimpleImputer(strategy=self.strategy)
        
    def transform_utterance(self, feature_matrix, utterance_id="UNKNOWN"):
        """
        Validates frame calculations. Resolves NaNs/Infs inline using most_frequent strategy metrics.
        
        Args:
            feature_matrix (np.ndarray): 2D Matrix (frames x channels)
            utterance_id (str): Logging trace metadata
        """
        # Convert infinite variations to true structural NaNs
        if np.isinf(feature_matrix).any():
            feature_matrix[np.isinf(feature_matrix)] = np.nan
            
        if not np.isnan(feature_matrix).any():
            return feature_matrix
            
        # Catch total corruption scenarios
        if np.isnan(feature_matrix).all():
            logger.warning(f"[CORRUPTED UTTERANCE] Entire matrix is NaN for {utterance_id}. Initializing with zeros.")
            return np.zeros_like(feature_matrix)
            
        try:
            # Reconstruct frame distributions safely
            healed_matrix = self.imputer.fit_transform(feature_matrix)
            return healed_matrix
        except Exception as e:
            logger.error(f"[IMPUTATION CRITICAL CRASH] Failed on item {utterance_id}: {str(e)}")
            # Fail-safe backup fallback
            return np.nan_to_num(feature_matrix, nan=0.0)