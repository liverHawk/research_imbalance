import yaml
import os
import numpy as np
import pandas as pd
from glob import glob
from lib.util import setup_logging
from classifier.improved_c45 import ImprovedC45
import mlflow


def train_process(model):
    log_path = os.path.join("logs", "train.log")
    logger = setup_logging(log_path)

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
    train_df = pd.concat([pd.read_csv(f, dtype=np.float32) for f in files], ignore_index=True)
    logger.info("Training model...")

    model.fit(train_df.drop("Label", axis=1), train_df["Label"])
    logger.info("Model training complete.")
    model.save(os.path.join("data", "models", "improved_c45_model.joblib"))
    mlflow.sklearn.log_model(
        sk_model=model,
        name="improved_c45_model",
        signature=mlflow.models.infer_signature(train_df.drop("Label", axis=1), model.predict(train_df.drop("Label", axis=1)))
    )


def main():
    os.makedirs(os.path.join("logs"), exist_ok=True)
    all_params = yaml.safe_load(open("params.yaml", "r"))
    mlflow_params = all_params["mlflow"]
    mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
    params = all_params["train"]
    print(params)
    os.makedirs(os.path.join("data", "models"), exist_ok=True)

    mlflow.set_experiment(f"{mlflow_params['experiment_name']}_train")
    mlflow.autolog()
    mlflow.start_run()
    mlflow.log_params({
        "sampling": all_params["prepare"]["sampling"],
        "sampling_method": all_params["prepare"]["sampling_params"]["method"],
    })
    train_process(params["classifier"])
    mlflow.end_run()


if __name__ == "__main__":
    main()
