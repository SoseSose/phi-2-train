# %%
import mlflow
from mlflow.entities import ViewType
import os
import json
import pandas as pd
from stgs import DB_PATH, Phi2_OUTPUT_FILE, EXPERIMENT_NAME
import pandas as pd


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

# %%


def calc_acc():
    tracking_uri = f"sqlite:///{DB_PATH}"
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        output_format="list",
        experiment_names=[EXPERIMENT_NAME],
    )


    for run in runs:
        # print(run.info.artifact_uri)
        location = mlflow.artifacts.download_artifacts(run.info.artifact_uri)
        with open(os.path.join(location, Phi2_OUTPUT_FILE)) as f:
            dic = json.load(f)
        df = pd.DataFrame(dic["data"], columns=dic["columns"])
        num_correct = calc_num_correct_from_df(df)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run.info.run_id):
            mlflow.log_metric('num correct', num_correct)

if __name__ == "__main__":
    calc_acc()

# %%
