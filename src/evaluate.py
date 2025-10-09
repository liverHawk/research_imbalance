import yaml
import os
import numpy as np
import pandas as pd
from lib.util import setup_logging
from classifier.improved_c45 import ImprovedC45
from glob import glob
import mlflow

import sklearn.metrics as metrics


def save_evaluation_results(predict_probs, test_labels: pd.Series):
    predict_actions = [prop.argmax() for prop in predict_probs]

    # predict_probs -> predict_actions vs test_label
    macro_prec = metrics.precision_score(
        test_labels, predict_actions, average='macro'
    )
    macro_f1 = metrics.f1_score(
        test_labels, predict_actions, average='macro'
    )
    emr = metrics.balanced_accuracy_score(test_labels, predict_actions)

    summary = {
        "macro_precision": macro_prec,
        "macro_f1": macro_f1,
        "balanced_accuracy": emr,
    }
    mlflow.log_metrics(summary)
    conf_matrix = metrics.confusion_matrix(test_labels, predict_actions)

    txt_path = os.path.join("evaluate", "confusion_matrix.csv")
    with open(txt_path, "w") as f:
        length = len(test_labels.unique())
        f.write(",".join([str(i) for i in range(length)]))
        f.write("\n")
        for index, row in enumerate(conf_matrix):
            f.write(f"{index},")
            f.write(",".join(str(i) for i in row))
            f.write("\n")
    normalized_path = os.path.join(
        "evaluate", "confusion_matrix_normalized.csv"
    )

    import matplotlib.pyplot as plt
    import seaborn as sns
    conf_matrix_normalize = metrics.confusion_matrix(
        test_labels, predict_actions, normalize='true'
    )
    # conf_matrix_normalize = metrics.confusion_matrix(
    #     predict_actions, test_labels, normalize='true'
    # )
    with open(normalized_path, "w") as f:
        length = len(test_labels.unique())
        f.write(",".join([str(i) for i in range(length)]))
        f.write("\n")
        for index, row in enumerate(conf_matrix_normalize):
            f.write(f"{index},")
            f.write(",".join(str(i) for i in row))
            f.write("\n")
    fig, ax = plt.subplots()
    sns.heatmap(
        conf_matrix_normalize.T, cmap="YlGn", ax=ax,
        linewidths=.5, linecolor='gray'
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    plt_path = os.path.join("evaluate", "confusion_matrix.png")
    fig.savefig(plt_path)
    mlflow.log_artifact(
        plt_path
    )
    plt.close(fig)


def load_mlflow_run_id():
    run_id_path = os.path.join("data", "models", "mlflow_run_id.txt")
    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            run_id = f.read().strip()
            return run_id
    else:
        return None


def evaluate_process():
    log_path = os.path.join("logs", "evaluate.log")
    logger = setup_logging(log_path)

    logger.info("Loading test data...")
    test_df = pd.concat(
        [pd.read_csv(f, dtype=np.float32) for f in glob(
            os.path.join("data", "prepared", "test", "*.csv.gz")
        )],
        ignore_index=True
    )

    logger.info("Loading model...")
    load_path = os.path.join("data", "models", "improved_c45_model.joblib")
    model = ImprovedC45(
        load_path=load_path
    )

    logger.info("Evaluating model...")
    predict_probs = model.predict_proba(test_df.drop("Label", axis=1))

    save_evaluation_results(predict_probs, test_df["Label"])

    # model_info = mlflow.sklearn.log_model(
    #     artifact_path="improved_c45_model_evaluated",
    #     sk_model=model,
    #     signature=mlflow.models.infer_signature(
    #         test_df.drop("Label", axis=1),
    #         model.predict(test_df.drop("Label", axis=1)))
    # )
    # mlflow.models.evaluate(
    #     model=model_info.model_uri,
    #     data=test_df,
    #     targets="Label",
    #     model_type="classifier",
    #     evaluators=["default"],
    # )


def main():
    params = yaml.safe_load(open("params.yaml", "r"))
    mlflow_params = params["mlflow"]
    evaluate_result_path = "evaluate"
    os.makedirs(evaluate_result_path, exist_ok=True)

    mlflow.set_experiment(f"{mlflow_params['experiment_name']}_evaluate")
    mlflow.set_tracking_uri(mlflow_params['tracking_uri'])

    mlflow.start_run()
    mlflow.log_params({
        "sampling": params["prepare"]["sampling"],
        "sampling_method": params["prepare"]["sampling_params"]["method"],
    })
    evaluate_process()
    mlflow.end_run()


if __name__ == "__main__":
    main()
