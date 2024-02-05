# %%
import mlflow
from mlruns_util import DB_PATH, EXPERIMENT_NAME, Phi2_OUTPUT_FILE
from mlflow.entities import ViewType
import json
import pandas as pd
import os
from arc_preprocess import make_2d_list_to_string, question_string_to_2d_list, string_to_two_d_list
from arc_visualize import plot_task
import numpy as np


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


def visualize_correct(df, name_key_ds):
    print(df.keys())
    for _, row in df.iterrows():
        name = row["question name"]
        data1 = name_key_ds[name]
        try:
            data = question_string_to_2d_list(data1)
        except:
            continue

        correct_answer =  np.array(data["true_out"])
        try:
            model_answer = np.array(string_to_two_d_list(row["answer"]))
        except:
            continue

        if np.all([correct_answer == model_answer]):
            print(data1["train"])
            plot_task(
                data["train"],
                [data["true_in"]],
                [data["true_out"]],
                model_answer=[model_answer],
                fold = 10
            )
        # break



def visualize():
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
        # print(df.head())
        train_or_eval = run.data.tags["train or Eval"]
        ds = make_2d_list_to_string(train_or_eval)
        # ds = questions_string_to_2d_list(ds)
        # print(ds[0])
        name_key_ds = {data["name"]: data for data in ds}
        visualize_correct(df, name_key_ds=name_key_ds)

        # num_correct = calc_num_correct_from_df(df)
        # visualize_correct(df)
        # mlflow.set_experiment(EXPERIMENT_NAME)
        # with mlflow.start_run(run.info.run_id):
        #     mlflow.log_metric("num correct", num_correct)


visualize()
# %%
