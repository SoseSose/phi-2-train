# %%
from data_processing.arc_preprocess import ArcTaskSet
from arc_visualize import plot_task
from utils import fix_random_seed
from architectures.phi2 import Phi2
from mlruns_util import MlflowRapper

if __name__ == "__main__":
    fix_random_seed()
    train_or_eval = "training"
    # train_or_eval = "evaluation"
    ds = ArcTaskSet().path_to_arc_task("data/"+train_or_eval)
    phi2 = Phi2("D:/models/phi2")
    mlflow_rapper = MlflowRapper()
    mlflow_rapper.evaluate_n_log(ds, phi2, train_or_eval)


import pytest


@pytest.mark.skip(reason="this test takes too long")
def calculate_num_collect():
    ds = ArcTaskSet().path_to_arc_task("data/training")
    model = Phi2("D:/models/phi2")
    mlflow_rapper = MlflowRapper()

    acc = mlflow_rapper.calculate_num_collect(ds, model)

    assert acc is not None


@pytest.mark.skip(reason="this test takes too long")
def test_evaluate_one():
    train_or_eval = "training"
    ds = ArcTaskSet().path_to_arc_task("data/"+train_or_eval)
    ds = ds[2:3]
    for data in ds:
        print(data.name)
    phi2 = Phi2("D:/models/phi2")
    # prompt = ds[0].to_str("example", "test") + "Answer the output"
    print(ds[0].name)
    prompt = ds[0].to_str("example", "test") 
    print(prompt)
    answer, token_num = phi2.get_token_num_and_answer(prompt)
    print("answer:")
    print(answer)

@pytest.mark.skip(reason="this test takes too long")
def test_plot_task_for_task_set():
    import numpy as np
    from pathlib import Path


    file_name = "test.png"
    ds = ArcTaskSet().path_to_arc_task("data/training")
    mock_answer = np.arange(25).reshape(5, 5)
    for data in ds[:2]:
        plot_task(
            train_inputs=data.train_inputs,
            train_outputs=data.train_outputs,
            test_inout=[data.test_input, data.test_output],
            candidate=data.candidate,
            model_answer=mock_answer,
            save_path=file_name,
        )
    Path(file_name).unlink()

