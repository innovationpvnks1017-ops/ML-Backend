"""Dataset loading and preprocessing utilities."""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """Loader for predefined datasets."""
    
    PREDEFINED_DATASETS = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }
    
    @staticmethod
    def _sklearn_to_df(ds, target_name="target"):
        """Convert sklearn Bunch dataset to DataFrame with target."""
        df = pd.DataFrame(ds.data, columns=[c for c in ds.feature_names])
        df[target_name] = ds.target
        return df
    
    @staticmethod
    def load(name: str) -> pd.DataFrame:
        """Load a predefined dataset by name."""
        name_norm = (name or "").strip().lower()
        
        if name_norm not in DatasetLoader.PREDEFINED_DATASETS:
            raise ValueError(
                f"Dataset '{name}' not supported. Available: {list(DatasetLoader.PREDEFINED_DATASETS.keys())}"
            )
        
        try:
            loader = DatasetLoader.PREDEFINED_DATASETS[name_norm]
            ds = loader()
            df = DatasetLoader._sklearn_to_df(ds, target_name="target")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{name}': {e}")
    
    @staticmethod
    def list_available() -> list:
        """List all available predefined datasets."""
        return list(DatasetLoader.PREDEFINED_DATASETS.keys())


class DataProcessor:
    """Preprocesses data for ML training."""
    
    @staticmethod
    def prepare_features(df: pd.DataFrame, target_col: str):
        """Extract and prepare features and target."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Select only numeric features and fill NaN
        X = X.select_dtypes(include=[np.number]).fillna(0)
        feature_names = list(X.columns)
        
        return X, y, feature_names
    
    @staticmethod
    def train_test_split_data(X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        return X_train, X_test, y_train, y_test
