# %%
from arc_preprocess import ArcTaskSet
from arc_visualize import plot_task
from utils import fix_random_seed
from phi2_model import Phi2
from mlruns_util import MlflowRapper

fix_random_seed()


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


# test_plot_task_for_task_set()

#%%
if __name__ == "__main__":

    phi2 = Phi2("D:/models/phi2")
    ds = ArcTaskSet().path_to_arc_task("data/training")
    for data in ds[:1]:
        question = data.to_str("example", "question") 
        # print(question)
        answer, token_num = phi2.get_token_num_and_answer(question)
        print(answer)

#%%
if __name__ == "__main__":

    phi2 = Phi2("D:/models/phi2")
    train_or_eval = "evaluation"
    ds = ArcTaskSet().path_to_arc_task("data/"+train_or_eval)
    # ds = ds[:10]
    mlflow_rapper = MlflowRapper()
    mlflow_rapper.evaluate_n_log(ds, phi2, train_or_eval)