import joblib
import pandas as pd
from pathlib import Path


class PredictionPipeline:

    def __init__(self):

        self.model = joblib.load(
            Path("artifacts/model_trainer/xgboost.joblib")
        )

        self.preprocessor = joblib.load(
            Path(
                "artifacts/data_transformation/preprocessor.joblib"
            )
        )

    def predict(self, data):

        # Transform raw user input
        transformed_data = self.preprocessor.transform(data)

        # Convert to DataFrame using the same feature names
        feature_names = self.preprocessor.get_feature_names_out()

        transformed_data = pd.DataFrame(
            transformed_data,
            columns=feature_names
        )

        # Predict
        prediction = self.model.predict(transformed_data)

        return prediction