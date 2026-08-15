import os
import json

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from mlProject import logger
from mlProject.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _resolve_tracking_uri(self):
        tracked_uri = os.getenv("MLFLOW_TRACKING_URI")

        if tracked_uri and not tracked_uri.startswith("file://") and "mlruns" not in tracked_uri.lower():
            return tracked_uri

        return "sqlite:///mlflow.db"

    def evaluate_model(self, model_path, model_name):

        test_data = pd.read_csv(
            self.config.test_data_path
        )

        target_column = self.config.target_column

        test_x = test_data.drop(
            columns=[target_column]
        )

        test_y = test_data[target_column]

        # -----------------------------
        # Load trained model
        # -----------------------------

        model = joblib.load(model_path)

        # -----------------------------
        # Predictions
        # -----------------------------

        predictions = model.predict(test_x)

        probabilities = model.predict_proba(test_x)[:, 1]

        # -----------------------------
        # Metrics
        # -----------------------------

        accuracy = accuracy_score(
            test_y,
            predictions
        )

        precision = precision_score(
            test_y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            test_y,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            test_y,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            test_y,
            probabilities
        )

        # -----------------------------
        # Confusion Matrix
        # -----------------------------

        cm = confusion_matrix(
            test_y,
            predictions
        )

        # -----------------------------
        # Classification Report
        # -----------------------------

        report = classification_report(
            test_y,
            predictions,
            zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        # -----------------------------
        # Create evaluation directory
        # -----------------------------

        os.makedirs(
            os.path.dirname(
                self.config.metric_file_name
            ),
            exist_ok=True
        )

        # -----------------------------
        # Save metrics locally
        # -----------------------------

        metrics_path = self.config.metric_file_name.replace(
            ".json",
            f"_{model_name}.json"
        )

        with open(metrics_path, "w") as f:
            json.dump(
                metrics,
                f,
                indent=4
            )

        # -----------------------------
        # MLflow Tracking
        # -----------------------------

        tracking_uri = self._resolve_tracking_uri()

        logger.info(
            f"Using MLflow tracking URI: {tracking_uri}"
        )

        mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(
            "loan_default_risk"
        )

        with mlflow.start_run(
            run_name=model_name
        ):

            # Parameters
            mlflow.log_param(
                "model_name",
                model_name
            )

            # Metrics
            mlflow.log_metric(
                "accuracy",
                accuracy
            )

            mlflow.log_metric(
                "precision",
                precision
            )

            mlflow.log_metric(
                "recall",
                recall
            )

            mlflow.log_metric(
                "f1_score",
                f1
            )

            mlflow.log_metric(
                "roc_auc",
                roc_auc
            )

            # -----------------------------
            # Confusion Matrix Artifact
            # -----------------------------

            confusion_matrix_path = os.path.join(
                self.config.root_dir,
                f"{model_name}_confusion_matrix.txt"
            )

            with open(
                confusion_matrix_path,
                "w"
            ) as f:
                f.write(str(cm))

            mlflow.log_artifact(
                confusion_matrix_path
            )

            # -----------------------------
            # Classification Report
            # -----------------------------

            report_path = os.path.join(
                self.config.root_dir,
                f"{model_name}_classification_report.txt"
            )

            with open(
                report_path,
                "w"
            ) as f:
                f.write(report)

            mlflow.log_artifact(
                report_path
            )

            # -----------------------------
            # Log Model
            # -----------------------------

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                serialization_format="cloudpickle"
            )

        # -----------------------------
        # Logging
        # -----------------------------

        logger.info(
            f"{model_name} evaluation completed."
        )

        logger.info(
            f"Accuracy: {accuracy:.4f}"
        )

        logger.info(
            f"Precision: {precision:.4f}"
        )

        logger.info(
            f"Recall: {recall:.4f}"
        )

        logger.info(
            f"F1 Score: {f1:.4f}"
        )

        logger.info(
            f"ROC-AUC: {roc_auc:.4f}"
        )

        return metrics