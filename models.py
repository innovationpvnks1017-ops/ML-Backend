"""ML Model wrapper classes for training and evaluation."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


class ModelFactory:
    """Factory for creating ML models."""
    
    MODELS = {
        "logisticregression": LogisticRegression,
        "logistic": LogisticRegression,
        "randomforest": RandomForestClassifier,
        "random-forest": RandomForestClassifier,
        "rf": RandomForestClassifier,
        "neuralnetwork": MLPClassifier,
        "neuralnet": MLPClassifier,
        "mlp": MLPClassifier,
        "svc": SVC,
        "svm": SVC,
    }
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize model name for lookup."""
        return (name or "").strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    
    @staticmethod
    def create(model_name: str, random_state: int = 42):
        """Create and return a model instance."""
        key = ModelFactory.normalize_name(model_name)
        ModelClass = ModelFactory.MODELS.get(key)
        
        if ModelClass is None:
            raise ValueError(f"Model '{model_name}' not supported. Supported: {list(ModelFactory.MODELS.keys())}")
        
        if ModelClass is SVC:
            return ModelClass(probability=True, random_state=random_state)
        elif ModelClass is MLPClassifier:
            return ModelClass(random_state=random_state, max_iter=500)
        elif ModelClass is RandomForestClassifier:
            return ModelClass(random_state=random_state)
        else:
            try:
                return ModelClass(max_iter=500, random_state=random_state)
            except TypeError:
                return ModelClass()


class ModelEvaluator:
    """Evaluates trained models and generates metrics."""
    
    @staticmethod
    def calculate_metrics(model, X_test, y_test, feature_names=None):
        """Calculate evaluation metrics for a trained model."""
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, "predict_proba"):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except Exception:
                y_pred_proba = None
        
        num_classes = len(np.unique(y_test))
        avg = "binary" if num_classes == 2 else "macro"
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        
        # Calculate ROC-AUC if applicable
        if y_pred_proba is not None:
            try:
                if num_classes == 2:
                    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
                        probs = y_pred_proba[:, 1]
                    else:
                        probs = np.ravel(y_pred_proba)
                    fpr, tpr, _ = roc_curve(y_test, probs)
                    roc_auc = auc(fpr, tpr)
                    metrics["roc_auc"] = float(roc_auc)
                else:
                    classes = np.unique(y_test)
                    y_test_bin = label_binarize(y_test, classes=classes)
                    if y_pred_proba.shape[1] == len(classes):
                        aucs = []
                        for i in range(len(classes)):
                            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                            aucs.append(float(auc(fpr, tpr)))
                        metrics["roc_auc_per_class"] = aucs
            except Exception:
                pass
        
        # Feature importance
        if feature_names:
            if hasattr(model, "feature_importances_"):
                metrics["feature_importance"] = {
                    str(f): float(i)
                    for f, i in zip(feature_names, model.feature_importances_)
                }
            elif hasattr(model, "coef_"):
                coefs = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
                metrics["feature_importance"] = {str(f): float(i) for f, i in zip(feature_names, coefs)}
        
        return metrics
