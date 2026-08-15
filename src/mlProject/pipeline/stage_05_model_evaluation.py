import os

from mlProject.config.configuration import ConfigurationManager
from mlProject.components.model_evaluation import ModelEvaluation
from mlProject import logger


STAGE_NAME = "Model evaluation stage"


class ModelEvaluationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        model_evaluation_config = (
            config.get_model_evaluation_config()
        )

        evaluation = ModelEvaluation(
            config=model_evaluation_config
        )

        model_directory = "artifacts/model_trainer"

        models = {
            "logistic_regression": os.path.join(
                model_directory,
                "logistic_regression.joblib"
            ),
            "random_forest": os.path.join(
                model_directory,
                "random_forest.joblib"
            ),
            "xgboost": os.path.join(
                model_directory,
                "xgboost.joblib"
            )
        }

        for model_name, model_path in models.items():

            logger.info(
                f"Evaluating {model_name}..."
            )

            evaluation.evaluate_model(
                model_path=model_path,
                model_name=model_name
            )


if __name__ == "__main__":

    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = ModelEvaluationTrainingPipeline()

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n"
            f"x==========x"
        )

    except Exception as e:

        logger.exception(e)

        raise e