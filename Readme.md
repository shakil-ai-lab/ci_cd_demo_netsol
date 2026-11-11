
# Breast Cancer Classification - CI Project

This project demonstrates Continuous Integration (CI)
using GitHub Actions.

## Dataset

Breast Cancer Wisconsin dataset from Scikit-learn.

- Samples: 569
- Features: 30
- Classes: 2

## Machine Learning Algorithm

Random Forest Classifier.

## Project Workflow

1. Load dataset
2. Split dataset
3. Train Random Forest
4. Evaluate model
5. Run automated tests
6. Execute tests automatically through GitHub Actions

## Install dependencies

pip install -r requirements.txt

## Train model

python train.py

## Run tests

pytest -v

## CI/CD

GitHub Actions automatically runs tests
whenever code is pushed or a pull request is created.
