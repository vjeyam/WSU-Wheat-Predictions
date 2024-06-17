"""
models package

This package contains modules for training, testing, and evaluating machine learning models for the wheat project.
It includes functionality for training a model, testing a model, and making time series predictions.

Modules:
    train: Functions for training machine learning models.
    test: Functions for testing trained machine learning models.
    evaluate_model: Functions for evaluating the performance of machine learning models.
    time_series_prediction: Functions for making time series predictions.

Example usage:
    from src.models import train, test, evaluate, predict_time_series

    # Train a model
    model = train(X_train, y_train, X_val, y_val, save_dir)

    # Test a model
    test_results = test(X_test, y_test, model, scaler)

    # Evaluate a model
    evaluation_metrics = evaluate(X_val, y_val, model, scaler)

    # Make time series predictions
    predictions = predict_time_series(model, scaler, X_future)
"""

from .train import train
from .test import test
from .evaluate_model import evaluate
from .time_series_prediction import predict_time_series
