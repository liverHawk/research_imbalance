import os
import yaml
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from ipaddress import ip_address as ip
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import imblearn.over_sampling as imblearn_os
from lib.util import setup_logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import mlflow

from lib.util import CICIDS2017, BASE


def adjust_labels(df, adjustment):
    labels = df["Label"].unique()

    # attempted: false -> change benign
    # combine_web_attack: true -> combine Web Attack
    # tool_separate: false -> combine DoS tools
    params = [
        adjustment["attempted"] is False,
        adjustment["combine_web_attack"] is True,
        adjustment["tool_separate"] is False,
    ]
    print(params)

    for label in labels:
        if params[0] and "Attempted" in label:
            df.loc[df["Label"] == label, "Label"] = "BENIGN"
            continue
        if params[1] and "Web Attack" in label:
            df.loc[df["Label"] == label, "Label"] = "Web Attack"
            continue
        if params[2] and "DoS " in label:
            df.loc[df["Label"] == label, "Label"] = "DoS"
            continue
    
    return df


def fast_process(df, type="normal"):
    if type == "normal":
        df = df.drop(CICIDS2017().get_delete_columns(), axis=1)
        df = df.drop(columns=['Attempted Category'])
    elif type == "full":
        df = df.drop(['Flow ID', 'Src IP', 'Attempted Category'], axis=1)
        # Timestamp→秒
        df['Timestamp'] = (
            pd.to_datetime(df['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
            .astype('int64') // 10**9
        )
        # IP文字列→整数
        df['Dst IP'] = df['Dst IP'].apply(lambda x: int(ip.IPv4Address(x)))
    # 欠損／無限大落とし
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def min_label_count(df):
    """
    Returns the minimum count of labels in the DataFrame.
    """
    return df["Label"].value_counts().min()


def oversampling(df, method, method_params):
    min_count = min_label_count(df)
    if min_count + 1 < method_params["neighbors"]:
        raise ValueError(
            f"Minimum label count {min_count} "
            f"is less than neighbors {method_params['neighbors']}"
        )

    if method == "None":
        return df
    elif method == "SMOTE":
        os_method = imblearn_os.SMOTE(
            k_neighbors=method_params["neighbors"],
            sampling_strategy=method_params["sampling_strategy"],
            random_state=method_params["seed"]
        )
    elif method == "ADASYN":
        os_method = imblearn_os.ADASYN(
            n_neighbors=method_params["neighbors"],
            sampling_strategy=method_params["sampling_strategy"],
            random_state=method_params["seed"]
        )
    else:
        raise ValueError(f"Unsupported oversampling method: {method}")

    X = df.drop("Label", axis=1)
    y = df["Label"]

    X_resampled, y_resampled = os_method.fit_resample(X, y)

    return pd.concat([X_resampled, y_resampled], axis=1)


def data_process(input_path, params):
    log_path = os.path.join("logs", "prepare.log")
    logger = setup_logging(log_path)

    logger.info("Starting data processing...")
    files = glob(f"{input_path}/*.csv")
    dfs = [fast_process(pd.read_csv(f)) for f in tqdm(files)]
    df = pd.concat(dfs, ignore_index=True)

    df = adjust_labels(df, params["adjustment"])
    df = df.dropna().dropna(axis=1, how='all')

    rename_dict = {
        k: v for k, v in zip(
            CICIDS2017().get_features_labels(),
            BASE().get_features_labels()
        )
    }
    df = df.rename(columns=rename_dict)

    logger.info("Label encoding...")
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=42,
        stratify=df["Label"]
    )

    test_value_counts = test_df["Label"].value_counts()

    with open("label.txt", "w") as f:
        for label, idx in zip(le.classes_, le.transform(le.classes_)):
            f.write(f"{idx}: {label}, test: {test_value_counts.get(idx, 0)}\n")

    logger.info("Oversampling the training data...")
    train_df = oversampling(
        train_df,
        method=params["oversampling"]["method"],
        method_params=params["oversampling"]["params"]
    )
    logger.info("Data processing completed.")

    return train_df, test_df


# ...existing code...
def save_split_csv(df, output_dir, output_prefix, chunk_size=100_000):
    arrays = []

    for i, start in enumerate(range(0, len(df), chunk_size)):
        path = os.path.join(output_dir, f"{output_prefix}_part{i+1}.csv.gz")
        arrays.append(
            [
                df.iloc[start:start+chunk_size],
                path
            ]
        )
    return arrays


def save_csv(df, path):
    df.to_csv(
        path,
        index=False,
        float_format="%.3f",
        compression="gzip"
    )


def multiprocess_save_csv(dfs, paths):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(save_csv, df, path) for df, path in zip(dfs, paths)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error during saving CSV files: {e}")
                raise e


def main():
    all_params = yaml.safe_load(open("params.yaml"))
    params = all_params["prepare"]

    mlflow.set_tracking_uri(all_params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(
        f"{all_params['mlflow']['experiment_name']}_prepare"
    )

    print(params)

    if len(sys.argv) != 2:
        print("Usage: python src/prepare.py <input_file_directory>")
        sys.exit(1)

    input = sys.argv[1]
    if not os.path.exists(input):
        print(f"Input file {input} does not exist.")
        sys.exit(1)

    prepare_dir = os.path.join(input, "..", "prepared")
    os.makedirs(os.path.join("logs"), exist_ok=True)
    os.makedirs(prepare_dir, exist_ok=True)
    os.makedirs(os.path.join(prepare_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(prepare_dir, "test"), exist_ok=True)

    output_train = os.path.join(prepare_dir, "train")
    output_test = os.path.join(prepare_dir, "test")

    dataset_path = os.path.join(input, params["dataset"])

    train_df, test_df = data_process(
        input_path=dataset_path,
        params=params
    )

    mlflow.start_run()
    mlflow.log_params(params)

    mlflow.log_metrics({
        "train_size": len(train_df),
        "test_size": len(test_df),
    })
    mlflow.log_dict(
        train_df["Label"].value_counts().to_dict(),
        "train_label_counts.json"
    )
    mlflow.log_dict(
        test_df["Label"].value_counts().to_dict(),
        "test_label_counts.json"
    )
    mlflow.end_run()

    # files = [
    #     [train_df, output_train, "train", 10_000_000],
    #     [test_df, output_test, "test", 500_000]
    # ]

    train = save_split_csv(train_df, output_train, "train", 10_000_000)
    test = save_split_csv(test_df, output_test, "test", 500_000)
    files = train + test

    try:
        multiprocess_save_csv(
            dfs=[df for df, _ in files],
            paths=[path for _, path in files]
        )
    except Exception as e:
        print(f"Error during saving CSV files: {e}")
        for file in files:
            save_csv(file[0], file[1])



if __name__ == "__main__":
    main()
