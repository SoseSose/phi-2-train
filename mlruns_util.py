import os
import sqlite3
import mlflow
import pandas as pd
from tqdm import tqdm
from mlflow.entities import ViewType

DB_PATH = "result/mlruns.db"
ARTIFACT_LOCATION = "result/artifacts"
EXPERIMENT_NAME = "test_mlflow"
Phi2_OUTPUT_FILE = "phi-2 select.json"


def calc_num_correct_from_df(df):
    num_correct = 0
    for _, row in df.iterrows():
        if row["answer"] in row["correct"]:
            num_correct += 1

    return num_correct


def test_calc_num_correct_from_df():
    data = {
        "answer": ["A", "B", "C"],
        "correct": [["A"], ["B"], ["D"]],
    }
    df = pd.DataFrame(data)
    num_correct = calc_num_correct_from_df(df)
    assert num_correct == 2


if __name__ == "__main__":
    test_calc_num_correct_from_df()


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
        self. experiment_id = experiment_id

    def evaluate_n_log(self, ds, model, train_or_eval):

        columns = ["input idetifer", "output"]
        data = {column: [] for column in columns}

        df = pd.DataFrame(data)

        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            mlflow.set_tag("train or Eval", train_or_eval)

            for i, data in tqdm(enumerate(ds[:400])):
                input_identifier = data["name"]
                output, token_num = model.get_token_num_and_answer(data)
                df.loc[i] = [input_identifier, output]

            mlflow.log_table(df, Phi2_OUTPUT_FILE)
            # dataframeのindexは保存されない。

    def calc_acc(self):
        tracking_uri = f"sqlite:///{DB_PATH}"
        mlflow.set_tracking_uri(tracking_uri)
        runs = mlflow.search_runs(
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            output_format="list",
            experiment_names=[EXPERIMENT_NAME],
        )

        for run in runs:
            print(run.info.run_id)
            location = mlflow.artifacts.download_artifacts(run.info.artifact_uri)
            with open(os.path.join(location, Phi2_OUTPUT_FILE)) as f:
                dic = json.load(f)
            df = pd.DataFrame(dic["data"], columns=dic["columns"])
            ds = make_2d_list_to_string("training")

            # num_correct = calc_num_correct_from_df(df)
            # visualize_correct(df)
            # mlflow.set_experiment(EXPERIMENT_NAME)
            # with mlflow.start_run(run.info.run_id):
            #     mlflow.log_metric("num correct", num_correct)


class Mock:
    def __init__(self) -> None:
        pass

    def build(self):
        pass
    
    def get_token_num_and_answer(self, question):
        return "mock", -1

def test_evalate_n_log():
    train_or_eval = "training"
    ds = ["aiueo" for _ in range(10)]
    mock = Mock()
    mlflow_rapper = MlflowRapper()
    mlflow_rapper.evaluate_n_log(ds, mock, train_or_eval)


# %%
# def visualize_correct(df):
#     for _, row in df.iterrows():
#         if row["answer"] in row["correct"]:
#             print(row["answer"])
#             two_d_list = string_to_two_d_list(row["answer"])
#             plot_some([two_d_list], "answer", show_num=True)


# %%




if __name__ == "__main__":
    calc_acc()

# %%