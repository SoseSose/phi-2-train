# %%
import torch
from data_preprocess import make_2d_list_to_string
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import mlflow
import sqlite3
import os
from stgs import DB_PATH, ARTIFACT_LOCATION, EXPERIMENT_NAME, Phi2_MAX_TOKENS, Phi2_OUTPUT_FILE 

class Phi2:
    def __init__(self) -> None:
        pass

    def build(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype="auto",
            # device_map="cuda:0",
            device_map="auto",
            cache_dir="D:/models",
            trust_remote_code=True,
        )
    
    def get_token_num_and_answer(self, question):
        try:
            with torch.no_grad():
                token_ids = self.tokenizer.encode(
                    question, add_special_tokens=False, return_tensors="pt"
                )
                token_num = len(token_ids[0])
                if token_num > Phi2_MAX_TOKENS:
                    answer = "token_num > {}".format(Phi2_MAX_TOKENS)
                else:
                    output_ids = self.model.generate(
                        token_ids.to(self.model.device),
                        temperature=0.2,
                        do_sample=True,
                        max_length=Phi2_MAX_TOKENS,
                    )
                    answer = self.tokenizer.decode(output_ids[0][token_ids.size(1) :])
                    answer = answer[: answer.find("\n\n")]

        except Exception as e:
            print("error")
            print(e)
            answer = token_num = "error"

        return answer, token_num

class Mock:
    def __init__(self) -> None:
        pass

    def build(self):
        pass
    
    def get_token_num_and_answer(self, question):
        return "mock", -1


def make_example_prompt(train_idx, inout):
    return f"example {train_idx}\n{inout['input']}\n->\n{inout['output']}\n\n"


def make_prompt(question):
    # prompt = "Instract: \n"
    prompt = ""
    prompt += "".join(
        make_example_prompt(train_idx, inout)
        for train_idx, inout in enumerate(question["train"])
    )
    prompt += "question\n" + question["true_in"] + "\n->\n"
    # prompt += "Output: "
    return prompt



def generate_ans(model, ds, train_or_eval):

    tracking_uri = f"sqlite:///{DB_PATH}"
    mlflow.set_tracking_uri(tracking_uri)


    tracking_uri = f"sqlite:///{DB_PATH}"
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:  
        experiment_id = mlflow.create_experiment(
            name=EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION
        )
    else: 
        experiment_id = experiment.experiment_id
    
    model.build()
    
    columns = ["question name", "question", "answer", "correct", "token_num"]
    data = {column: [] for column in columns}

    df = pd.DataFrame(data)
    df.set_index("question name", inplace=True)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tag("train or Eval", train_or_eval)

        for data in tqdm(ds[:400]):
            question = make_prompt(data)
            correct = data["true_out"]
            answer, token_num = model.get_token_num_and_answer(question)
            df.loc[data["name"]] = [question, answer, correct, token_num]

        mlflow.log_table(df, Phi2_OUTPUT_FILE)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH) 
    train_or_eval = "training"
    ds = make_2d_list_to_string(train_or_eval)
    ds = ds[:5]
    mock = Mock()
    generate_ans(mock, ds, train_or_eval)

#%%

if __name__ == "__main__":
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH) 

    phi2 = Phi2()
    train_or_eval = "training"
    ds = make_2d_list_to_string(train_or_eval)
    generate_ans(Phi2, ds, train_or_eval)

    phi2 = Phi2()
    train_or_eval = "evaluation"
    ds = make_2d_list_to_string(train_or_eval)
    generate_ans(Phi2, ds, train_or_eval)
