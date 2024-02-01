import mlflow
from mlruns_util import DB_PATH, EXPERIMENT_NAME, Phi2_OUTPUT_FILE 
from mlflow.entities import ViewType
import json
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


# %%
# def visualize_correct(df):
#     for _, row in df.iterrows():
#         if row["answer"] in row["correct"]:
#             print(row["answer"])
#             two_d_list = string_to_two_d_list(row["answer"])
#             plot_some([two_d_list], "answer", show_num=True)

def visualize_correct():
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