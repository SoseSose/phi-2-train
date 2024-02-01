import os
import sqlite3
import mlflow
import pandas as pd
from tqdm import tqdm

DB_PATH = "result/mlruns.db"
ARTIFACT_LOCATION = "result/artifacts"
EXPERIMENT_NAME = "test_mlflow"
Phi2_OUTPUT_FILE = "phi-2 select.json"


class MlflowRapper:
    def __init__(self) -> None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)

        tracking_uri = f"sqlite:///{DB_PATH}"
        mlflow.set_tracking_uri(tracking_uri)

        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION
            )
        else:
            experiment_id = experiment.experiment_id
        self.experiment_id = experiment_id

    def evaluate_n_log(self, ds, model, train_or_eval):
        columns = ["input idetifer", "output", "token num"]
        data = {column: [] for column in columns}

        df = pd.DataFrame(data)

        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            mlflow.set_tag("train or Eval", train_or_eval)

            for i, data in tqdm(enumerate(ds)):
                input_identifier = data["name"]
                question = data["question"]
                output, token_num = model.get_token_num_and_answer(question)
                df.loc[i] = [input_identifier, output, token_num]

            mlflow.log_table(df, Phi2_OUTPUT_FILE)
            run_id = run.info.run_id
            # dataframeのindexは保存されない。

        return run_id


class Mock:
    def __init__(self) -> None:
        pass

    def build(self):
        pass

    def get_token_num_and_answer(self, question):
        return "mock", -1


def test_evalate_n_log():
    train_or_eval = "training"
    ds = [{"name": "aiueo", "question": "test"} for _ in range(10)]
    mock = Mock()
    mlflow_rapper = MlflowRapper()
    run_id = mlflow_rapper.evaluate_n_log(ds, mock, train_or_eval)
    mlflow.delete_run(run_id)

