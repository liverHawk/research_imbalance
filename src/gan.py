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
    # train_df = oversampling(
    #     train_df,
    #     method=params["oversampling"]["method"],
    #     method_params=params["oversampling"]["params"]
    # )
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


def change_columns(df):
    """
    Seriesを一気にフォーマットして文字列にし、各文字を個別の列として保存
    """
    new_df = df.copy()
    
    # Labelカラムは除外
    feature_columns = [
        {"type": "Destination Port", "max": 65535},
        {"type": "Protocol", "max": 255}
    ]

    for col in feature_columns:
        col_name = col["type"]
        
        # 数値型でない場合はスキップ
        if not pd.api.types.is_numeric_dtype(df[col_name]):
            continue
            
        # 整数値に変換（小数点以下は切り捨て）
        int_values = df[col_name].astype(int)

        # 最大値から必要な桁数を決定
        max_val = col["max"]
        max_val_binary = format(max_val, 'b')
        num_digits = len(max_val_binary)
        
        # Series全体を一気に2進数文字列にフォーマット（ゼロパディング）  
        binary_strings = int_values.apply(lambda x: format(x, f'0{num_digits}b'))
        
        # 各文字位置を個別の列として作成
        for char_pos in range(num_digits):
            bit_col_name = f"{col_name}_{char_pos}"
            # 各文字列の指定位置の文字を取得して整数に変換
            new_df[bit_col_name] = binary_strings.str[char_pos].astype(int)
        
        # 元の列を削除
        new_df = new_df.drop(columns=[col_name])
        
        print(f"列 '{col_name}' を {num_digits} 文字に分割しました")
    
    return new_df

def main():
    all_params = yaml.safe_load(open("params.yaml"))
    params = all_params["prepare"]

    mlflow.set_tracking_uri(all_params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(
        f"{all_params['mlflow']['experiment_name']}_gan"
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

    # train = save_split_csv(train_df, output_train, "train", 10_000_000)
    # test = save_split_csv(test_df, output_test, "test", 500_000)
    # files = train + test

    # try:
    #     multiprocess_save_csv(
    #         dfs=[df for df, _ in files],
    #         paths=[path for _, path in files]
    #     )
    # except Exception as e:
    #     print(f"Error during saving CSV files: {e}")
    #     for file in files:
    #         save_csv(file[0], file[1])
    
    # 列の値を2進数化して各ビット位置を個別の列として保存
    print("訓練データの列を2進数化しています...")
    train_df = change_columns(train_df)
    path = os.path.join(output_train, "../binary")
    binary_train = save_split_csv(train_df, output_train, "train", 100_000)
    try:
        multiprocess_save_csv(
            dfs=[df for df, _ in binary_train],
            paths=[path for _, path in binary_train]
        )
    except Exception as e:
        print(f"Error during saving CSV files: {e}")
        for df, path in binary_train:
            save_csv(df, path)


if __name__ == "__main__":
    main()
