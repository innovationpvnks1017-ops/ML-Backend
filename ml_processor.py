"""Main ML processor that orchestrates training and evaluation."""
from backend.data_loader import DatasetLoader, DataProcessor
from backend.models import ModelFactory, ModelEvaluator


class MLProcessor:
    """Orchestrates ML workflow: load data, train model, evaluate."""
    
    def __init__(self, dataset_name: str, model_name: str, test_size: float = 0.2, random_state: int = 42):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        
        self.df = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.feature_names = None
        self.target_col = "target"
        self.model = None
    
    def load_dataset(self):
        """Load the specified dataset."""
        self.df = DatasetLoader.load(self.dataset_name)
    
    def prepare_data(self):
        """Prepare features and target from dataset."""
        X, y, self.feature_names = DataProcessor.prepare_features(self.df, self.target_col)
        self.X_train, self.X_test, self.y_train, self.y_test = DataProcessor.train_test_split_data(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
    
    def train_model(self):
        """Create and train the model."""
        self.model = ModelFactory.create(self.model_name, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        
        metrics = ModelEvaluator.calculate_metrics(
            self.model, self.X_test, self.y_test, feature_names=self.feature_names
        )
        return metrics
    
    def run_experiment(self):
        """Execute full ML pipeline."""
        self.load_dataset()
        self.prepare_data()
        self.train_model()
        metrics = self.evaluate_model()
        
        return {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "test_size": self.test_size,
            "metrics": metrics,
        }
