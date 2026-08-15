import os
from pathlib import Path

import pandas as pd

from mlProject.config.configuration import ConfigurationManager


def test_configuration_manager_uses_local_mlruns_uri():
    config = ConfigurationManager()
    model_eval = config.get_model_evaluation_config()
    assert "mlruns" in model_eval.mlflow_uri


def test_model_artifacts_exist_after_training():
    root = Path("artifacts/model_trainer")
    required = [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
    ]
    for file_name in required:
        assert (root / file_name).exists(), f"Missing {file_name}"


def test_processed_training_data_frames_are_valid():
    train_path = Path("artifacts/data_transformation/train.csv")
    test_path = Path("artifacts/data_transformation/test.csv")
    assert train_path.exists(), "Training dataset missing"
    assert test_path.exists(), "Test dataset missing"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert "Default" in train_df.columns
    assert "Default" in test_df.columns
    assert not train_df.empty
    assert not test_df.empty
