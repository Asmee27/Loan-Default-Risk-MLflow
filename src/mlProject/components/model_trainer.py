import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        train_data = pd.read_csv(self.config.train_data_path)

        target_column = self.config.target_column

        train_x = train_data.drop(columns=[target_column])
        train_y = train_data[target_column]

        params = self.config.model_params

        logger.info(f"Training data shape: {train_x.shape}")
        logger.info(f"Target column: {target_column}")

        # -----------------------------
        # 1. Logistic Regression
        # -----------------------------

        logistic_params = params.logistic_regression

        logistic_model = LogisticRegression(
            C=logistic_params.C,
            max_iter=logistic_params.max_iter,
            class_weight=logistic_params.class_weight,
            random_state=42
        )

        logistic_model.fit(train_x, train_y)

        joblib.dump(
            logistic_model,
            os.path.join(
                self.config.root_dir,
                "logistic_regression.joblib"
            )
        )

        logger.info(
            "Balanced Logistic Regression model trained successfully."
        )

        # -----------------------------
        # 2. Random Forest
        # -----------------------------

        random_forest_params = params.random_forest

        random_forest_model = RandomForestClassifier(
            n_estimators=random_forest_params.n_estimators,
            max_depth=random_forest_params.max_depth,
            min_samples_split=random_forest_params.min_samples_split,
            class_weight=random_forest_params.class_weight,
            random_state=random_forest_params.random_state,
            n_jobs=1
        )

        random_forest_model.fit(train_x, train_y)

        joblib.dump(
            random_forest_model,
            os.path.join(
                self.config.root_dir,
                "random_forest.joblib"
            )
        )

        logger.info(
            "Balanced Random Forest model trained successfully."
        )

        # -----------------------------
        # 3. XGBoost
        # -----------------------------

        xgboost_params = params.xgboost

        xgboost_model = XGBClassifier(
            n_estimators=xgboost_params.n_estimators,
            max_depth=xgboost_params.max_depth,
            learning_rate=xgboost_params.learning_rate,
            subsample=xgboost_params.subsample,
            colsample_bytree=xgboost_params.colsample_bytree,
            scale_pos_weight=xgboost_params.scale_pos_weight,
            random_state=xgboost_params.random_state,
            eval_metric="logloss",
            n_jobs=1
        )

        xgboost_model.fit(train_x, train_y)

        joblib.dump(
            xgboost_model,
            os.path.join(
                self.config.root_dir,
                "xgboost.joblib"
            )
        )

        # Save XGBoost as the production model
        joblib.dump(
            xgboost_model,
            os.path.join(
                self.config.root_dir,
                "model.joblib"
            )
        )

        logger.info(
            "XGBoost saved as the production model."
        )

        logger.info(
            "Balanced XGBoost model trained successfully."
        )

        logger.info(
            "All balanced models trained successfully."
        )