from sklearn import datasets
import pandas as pd


class DatasetLoader:
    @staticmethod
    def load(name: str):
        name = name.lower().replace("-", "").replace("_", "")

        if name == "iris":
            data = datasets.load_iris()
        elif name == "wine":
            data = datasets.load_wine()
        elif name == "breastcancer":
            data = datasets.load_breast_cancer()
        elif name == "diabetes":
            data = datasets.load_diabetes()
        elif name == "heartdisease":
            # Example: synthetic heart dataset
            df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/heart.csv")
            X = df.drop("target", axis=1)
            y = df["target"]
            return X, y, X.columns.tolist()
        else:
            raise ValueError(f"Unknown dataset '{name}'")

        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        return X, y, data.feature_names
