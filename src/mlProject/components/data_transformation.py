import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):

        data = pd.read_csv(self.config.data_path)

        logger.info(f"Loaded dataset with shape: {data.shape}")

        # Remove identifier column
        if "LoanID" in data.columns:
            data = data.drop(columns=["LoanID"])

        target_column = "Default"

        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Identify numerical and categorical columns
        numerical_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            include=["object"]
        ).columns.tolist()

        logger.info(f"Numerical features: {numerical_features}")
        logger.info(f"Categorical features: {categorical_features}")

        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    numerical_features
                ),
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                    categorical_features
                )
            ]
        )

        # Stratified split to preserve default/non-default ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # Fit only on training data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Convert transformed arrays to DataFrames
        feature_names = preprocessor.get_feature_names_out()

        X_train_transformed = pd.DataFrame(
            X_train_transformed,
            columns=feature_names
        )

        X_test_transformed = pd.DataFrame(
            X_test_transformed,
            columns=feature_names
        )
        os.makedirs(self.config.root_dir, exist_ok=True)

        # Save the fitted preprocessor for future inference
        preprocessor_path = os.path.join(
            self.config.root_dir,
            "preprocessor.joblib"
        )

        joblib.dump(preprocessor, preprocessor_path)

        logger.info(
            f"Preprocessor saved successfully: {preprocessor_path}"
        )

        # Add target column
        X_train_transformed[target_column] = y_train.reset_index(drop=True)
        X_test_transformed[target_column] = y_test.reset_index(drop=True)


        # Save processed datasets
        X_train_transformed.to_csv(
            os.path.join(self.config.root_dir, "train.csv"),
            index=False
        )

        X_test_transformed.to_csv(
            os.path.join(self.config.root_dir, "test.csv"),
            index=False
        )

        logger.info(
            f"Training data saved: {X_train_transformed.shape}"
        )

        logger.info(
            f"Testing data saved: {X_test_transformed.shape}"
        )

        print(f"Training data shape: {X_train_transformed.shape}")
        print(f"Testing data shape: {X_test_transformed.shape}")