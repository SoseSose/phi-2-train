# %%
from logical_op_ds_make import load_logical_tasks
from data_processing.arc_preprocess import ArcTask, ArcTaskSet
from phi2_model import Phi2
from torch.utils.data import DataLoader
from torch.optim import SGD
from transformers import get_scheduler
import torch
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from utils import fix_random_seed

fix_random_seed()

##!くそ長い時間がかかるのでlogical_opに置き換える。
class Model:
    def __init__(self) -> None:

        device = torch.device("cuda:0") 
        self.phi2 = Phi2("D:/models/phi2", device=device)
        self.max_length = int(self.phi2.MAX_TOKENS * 3 / 5)
    

    def tokenize_function(self,examples):

        token_ids = self.phi2.tokenizer.encode(
            examples,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation="only_first",
            return_tensors="pt",
        )
        token_len = len(token_ids[0])
        return token_ids, token_len

    def train_ds(self, ds):
        train_ds = [self.tokenize_function(ds["train"][i]) for i in range(1000)]
        train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=1, drop_last=True)  # type: ignore

        optimizer = SGD(self.phi2.model.parameters(), lr=5e-5)


        num_epochs = 3
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        # progress_bar = tqdm(range(num_training_steps))
        logits = 51200

        self.phi2.model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                question, ans = batch
                outputs = self.phi2.model(question.to(self.device))
                outputs_reshape = outputs.logits.view(-1, outputs.logits.shape[-1])
                ans = ans.reshape((ans.shape[-1],)).to(self.device)
                loss = loss_fn(outputs_reshape, ans)

                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


    def eval_ds(self, ds):
        def split_q_n_a(examples:ArcTask):
            question = examples.to_str("example", "test", show_test_out=False)
            answer = examples.test_output.to_str("", ",")
            return question, answer
        
        ds = [split_q_n_a(ds[i]) for i in range(1000)]
        eval_dataloader = DataLoader(ds, batch_size=1, drop_last=True)  

        for batch in eval_dataloader:
            question, ans = batch
            outputs = self.phi2.model(question.to(self.device))
            outputs_reshape = outputs.logits.view(-1, outputs.logits.shape[-1])
            ans = ans.reshape((ans.shape[-1],)).to(self.device)
            loss = loss_fn(outputs_reshape, ans)

    



import pytest
@pytest.mark.skip(reason="this test takes too long")
def test_main():
    mainclass = Model()
    eval_ds = ()
    before_train_acc = Model.eval_ds(eval_ds)
    train_ds = load_logical_tasks(Path("data/logical_op/train"))

    mainclass.train_ds(LogicalOpDs)

    acc = Model.eval_ds(eval_ds)
    assert acc > before_train_acc

if __name__ == "__main__":
    test_main()


