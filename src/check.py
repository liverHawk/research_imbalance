import yaml
import os
import sys
import pandas as pd
import mlflow
from tqdm import tqdm
from glob import glob
from lib.util import setup_logging


def setup(params):
    os.makedirs("check", exist_ok=True)
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(f'{params["mlflow"]["experiment_name"]}_check')

def save_data_info(dataset_path):
    mlflow.start_run()
    log_path = os.path.join("logs", "check.log")
    logger = setup_logging(log_path)
    logger.info(f"Dataset: {dataset_path}")

    files = glob(os.path.join(dataset_path, "*"))
    dfs = [
        pd.read_csv(file, usecols=["Label"]) for file in tqdm(files) if file.endswith(".csv")
    ]
    logger.info(f"Number of files: {len(dfs)}")
    df = pd.concat(dfs, ignore_index=True)

    logger.info("Saving data information...")
    labels = df["Label"].value_counts()
    labels.to_csv(os.path.join("check", "label_distribution.csv"))
    mlflow.log_artifact(
        os.path.join("check", "label_distribution.csv"),
        artifact_path="check"
    )

    # columns = df.columns
    # with open(os.path.join("check", "columns.txt"), "w") as f:
    #     for col in columns:
    #         f.write(f"{col}\n")
    # mlflow.log_artifact(
    #     os.path.join("check", "columns.txt"),
    #     artifact_path="check"
    # )
    mlflow.end_run()


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/check.py <input_directory>")
        sys.exit(1)
    input = sys.argv[1]
    
    params = yaml.safe_load(open("params.yaml"))
    dataset = params["prepare"]["dataset"]
    print(f"Dataset: {dataset}")

    setup(params)
    
    save_data_info(
        os.path.join(input, dataset)
    )


if __name__ == "__main__":
    main()