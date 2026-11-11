
from pathlib import Path
import joblib

from src.data import load_dataset
from src.model import train_and_evaluate


def main():

    # Step 1: Load dataset
    X, y = load_dataset()

    print("Dataset shape:", X.shape)
    print("Target shape:", y.shape)

    # Step 2: Train and evaluate model
    model, accuracy, X_test, y_test, predictions = train_and_evaluate(X, y)

    print("Model Accuracy:", accuracy)

    # Step 3: Save trained model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(
        model,
        models_dir / "breast_cancer_model.joblib"
    )

    print("Model saved successfully!")


if __name__ == "__main__":
    main()
