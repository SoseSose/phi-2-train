# %%
import mlflow
from mlruns_util import DB_PATH, EXPERIMENT_NAME, Phi2_OUTPUT_FILE
from mlflow.entities import ViewType
import json
import pandas as pd
import os
from arc_visualize import plot_task
from arc_preprocess import ArcTaskSet, str_to_arc_image, ArcImage


def visualize_correct(df, train_or_eval):

    ds = ArcTaskSet().path_to_arc_task("data/" + train_or_eval)
    name_key_ds = {data.name: data for data in ds}

    for _, row in df.iterrows():
        name = row["input idetifer"]
        data = name_key_ds[name]

        correct_ans = data.test_output
        if isinstance(correct_ans, ArcImage):
            correct_ans = correct_ans.to_str("", "\n")

        model_answer = row["output"].split("\n\n")[0]

        file_path = f"result//arc_phi2_correct//{train_or_eval}//{name}.png" 
        if correct_ans == model_answer:
            model_answer = str_to_arc_image(model_answer)
            plot_task(
                train_inputs=data.train_inputs,
                train_outputs=data.train_outputs,
                test_inout=[data.test_input, data.test_output],
                model_answer=model_answer,
                save_path=file_path,
            )


def visualize():
    analyze_id = [
        "eadc0e6e32c34bceb5dacd75b839b6a9",
        "ac815d2c58c54a64bd2e4226a5456aa2",
    ]
    tracking_uri = f"sqlite:///{DB_PATH}"
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        output_format="list",
        experiment_names=[EXPERIMENT_NAME],
    )

    for run in runs:
        if run.info.run_id in analyze_id:
            location = mlflow.artifacts.download_artifacts(run.info.artifact_uri)
            print(location)
            with open(os.path.join(location, Phi2_OUTPUT_FILE)) as f:
                dic = json.load(f)
            df = pd.DataFrame(dic["data"], columns=dic["columns"])
            print(df.keys())
            train_or_eval = run.data.tags["train or Eval"]
            visualize_correct(df, train_or_eval)

# %%
if __name__ == "__main__":
    visualize()
