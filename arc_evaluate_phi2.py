# %%
from arc_preprocess import make_2d_list_to_string, make_prompt
from utils import fix_random_seed
from phi2_model import Phi2
from mlruns_util import MlflowRapper

fix_random_seed()


# %%

if __name__ == "__main__":

    phi2 = Phi2("D:/models/phi2")
    train_or_eval = "training"
    ds = make_2d_list_to_string(train_or_eval)
    ds = [make_prompt(data) for data in ds]
    mlflow_rapper = MlflowRapper
    mlflow_rapper.evaluate_n_log(ds,phi2, train_or_eval)

    phi2 = Phi2("D:/models/phi2")
    train_or_eval = "evaluation"
    ds = make_2d_list_to_string(train_or_eval)
    ds = [make_prompt(data) for data in ds]
    generate_ans(phi2, ds, train_or_eval)
