import yaml
import os
import pandas as pd
from glob import glob
from lib.util import setup_logging
from classifier.improved_c45 import ImprovedC45
import mlflow


@mlflow.trace
def train_process(model, mlflow_params):
    log_path = os.path.join("logs", "train.log")
    logger = setup_logging(log_path)

    logger.info(f"MLflow tracking URI: {mlflow_params['tracking_uri']}")
    mlflow.set_experiment(mlflow_params['experiment_name'])
    mlflow.autolog()
    mlflow.start_run()

    if model["model"] == "ImprovedC45":
        logger.info("Using ImprovedC45 classifier.")
        model = ImprovedC45(
            max_depth=model["params"]["max_depth"],
        )
    else:
        raise ValueError(
            f"Unsupported classifier: {model['model']}"
        )

    logger.info("Loading training data...")
    files = glob(os.path.join("data", "prepared", "train", "*.csv.gz"))
    train_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    logger.info("Training model...")

    model.fit(train_df.drop("Label", axis=1), train_df["Label"])
    logger.info("Model training complete.")
    model.save(os.path.join("data", "models", "improved_c45_model.joblib"))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="improved_c45_model"
    )

    mlflow.end_run()


def main():
    os.makedirs(os.path.join("logs"), exist_ok=True)
    all_params = yaml.safe_load(open("params.yaml", "r"))
    mlflow_params = all_params["mlflow"]
    mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
    params = all_params["train"]
    print(params)
    os.makedirs(os.path.join("data", "models"), exist_ok=True)

    train_process(params["classifier"], mlflow_params)


if __name__ == "__main__":
    main()
