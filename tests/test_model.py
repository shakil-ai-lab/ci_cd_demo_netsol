
import numpy as np

from src.data import load_dataset
from src.model import build_model, train_and_evaluate


def test_model_can_be_created():

    model = build_model()

    assert model is not None


def test_model_trains_and_predicts():

    X, y = load_dataset()

    model, accuracy, X_test, y_test, predictions = train_and_evaluate(X, y)

    # Accuracy should be between 0 and 1
    assert 0 <= accuracy <= 1

    # Minimum expected accuracy for this exercise
    assert accuracy >= 0.90

    # Prediction count should match test samples
    assert len(predictions) == len(y_test)

    # Predictions should contain valid classes
    assert set(np.unique(predictions)).issubset({0, 1})


def test_prediction_shape():

    X, y = load_dataset()

    model, accuracy, X_test, y_test, predictions = train_and_evaluate(X, y)

    assert predictions.shape == y_test.shape
