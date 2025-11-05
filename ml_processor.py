import io
import base64
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datasets import DatasetLoader
from models import ModelFactory, ModelEvaluator


class MLProcessor:
    def __init__(self, dataset_name, model_name, test_size=0.2, random_state=42, visualizations=None):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        self.visualizations = visualizations or []

    def run_experiment(self):
        # Load dataset
        X, y, feature_names = DatasetLoader.load(self.dataset_name)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Train model
        model = ModelFactory.create(self.model_name, random_state=self.random_state)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = ModelEvaluator.calculate_metrics(model, X_test, y_test, feature_names)

        # Add visualizations
        visuals = {}
        for viz in self.visualizations:
            try:
                visuals[viz] = self._generate_visualization(viz, metrics)
            except Exception:
                visuals[viz] = None

        return {"model": self.model_name, "dataset": self.dataset_name, "metrics": metrics, "visualizations": visuals}

    def _generate_visualization(self, viz, metrics):
        plt.figure()
        if viz == "confusion-matrix" and "confusion_matrix" in metrics:
            import seaborn as sns
            sns.heatmap(metrics["confusion_matrix"], annot=True, cmap="Blues")
            plt.title("Confusion Matrix")
        elif viz == "roc-curve" and "roc_auc" in metrics:
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title(f"ROC Curve (AUC={metrics['roc_auc']:.2f})")
        elif viz == "feature-importance" and "feature_importance" in metrics:
            items = metrics["feature_importance"].items()
            plt.bar([k for k, _ in items], [v for _, v in items])
            plt.xticks(rotation=45)
            plt.title("Feature Importance")
        else:
            plt.text(0.5, 0.5, "Visualization not available", ha="center")

        # Encode image
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{encoded}"
