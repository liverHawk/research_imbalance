import yaml
import os
import sys
import pandas as pd
from lib.util import setup_logging
from classifier.improved_c45 import ImprovedC45
from dvclive import Live
from glob import glob
import mlflow
from mlflow.models import infer_signature

import sklearn.metrics as metrics


def save_evaluation_results(live: Live, predict_probs, test_labels: pd.Series):
    predict_actions = [prop.argmax() for prop in predict_probs]

    # predict_probs -> predict_actions vs test_label
    macro_prec = metrics.precision_score(test_labels, predict_actions, average='macro')
    macro_f1 = metrics.f1_score(test_labels, predict_actions, average='macro')
    emr = metrics.balanced_accuracy_score(test_labels, predict_actions)

    live.summary = {
        "macro_precision": macro_prec,
        "macro_f1": macro_f1,
        "balanced_accuracy": emr,
    }
    mlflow.log_metrics(live.summary)

    live.log_sklearn_plot(
        "confusion_matrix",
        test_labels,
        predict_actions,
        title="Confusion Matrix",
    )
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
    normalized_path = os.path.join("evaluate", "confusion_matrix_normalized.csv")

    import matplotlib.pyplot as plt
    import seaborn as sns
    conf_matrix_normalize = metrics.confusion_matrix(test_labels, predict_actions, normalize='true')
    # conf_matrix_normalize = metrics.confusion_matrix(predict_actions, test_labels, normalize='true')
    with open(normalized_path, "w") as f:
        length = len(test_labels.unique())
        f.write(",".join([str(i) for i in range(length)]))
        f.write("\n")
        for index, row in enumerate(conf_matrix_normalize):
            f.write(f"{index},")
            f.write(",".join(str(i) for i in row))
            f.write("\n")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix_normalize.T, cmap="YlGn", ax=ax, linewidths=.5, linecolor='gray')
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    live.log_image(
        "confusion_matrix_image.png",
        fig,
    )

def load_mlflow_run_id():
    run_id_path = os.path.join("data", "models", "mlflow_run_id.txt")
    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            run_id = f.read().strip()
            return run_id
    else:
        return None

def evaluate_process(evaluate_path, mlflow_params):
    mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
    mlflow.set_experiment(mlflow_params['experiment_name'])
    mlflow.start_run()

    log_path = os.path.join("logs", "evaluate.log")
    logger = setup_logging(log_path)

    logger.info("Loading test data...")
    test_df = pd.concat([pd.read_csv(f) for f in glob(os.path.join("data", "prepared", "test", "*.csv.gz"))], ignore_index=True)
    
    logger.info("Loading model...")
    load_path = os.path.join("data", "models", "improved_c45_model", "MLmodel")
    model_load = mlflow.sklearn.load_model(load_path)
    model = ImprovedC45(
        load_path=load_path
    )
    
    logger.info("Evaluating model...")
    predict_probs = model.predict_proba(test_df.drop("Label", axis=1))

    with Live(evaluate_path) as live:
        save_evaluation_results(live, predict_probs, test_df["Label"])

    mlflow.models.evaluate(
        model=model_load.model_uri,
        data=test_df,
        targets="Label",
        model_type="classifier",
        evaluators=["default"]
    )

    mlflow.end_run()


def main():
    params = yaml.safe_load(open("params.yaml", "r"))
    mlflow_params = params["mlflow"]
    evaluate_result_path = os.path.join("evaluate")

    evaluate_process(evaluate_result_path, mlflow_params)


if __name__ == "__main__":
    main()