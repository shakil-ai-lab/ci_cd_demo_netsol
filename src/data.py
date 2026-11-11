
from sklearn.datasets import load_breast_cancer


def load_dataset():
    """
    Load Breast Cancer dataset.

    Returns:
        X: Input features
        y: Target labels
    """

    dataset = load_breast_cancer()

    X = dataset.data
    y = dataset.target

    return X, y
