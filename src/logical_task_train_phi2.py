# %%
from datasets import load_dataset
from phi2_model import Phi2
from torch.utils.data import DataLoader
from torch.optim import SGD
from transformers import get_scheduler
import torch
from tqdm import tqdm
import torch.nn as nn

def main():
    ##!くそ長い時間がかかるのでlogical_opに置き換える。
    dataset = load_dataset("yelp_review_full")
    print("load dataset")
    device = torch.device("cuda:0") 
    phi2 = Phi2("D:/models/phi2", device=device)
    max_length = int(phi2.MAX_TOKENS * 3 / 5)

    def tokenize_function(examples):

        token_ids = phi2.tokenizer.encode(
            examples["text"],
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation="only_first",
            return_tensors="pt",
        )
        token_len = len(token_ids[0])
        split_len = int(token_len /2)
        question, ans = token_ids[0][:split_len], token_ids[0][split_len:]
        return question, ans
        # return token_ids


    def test_tokenize_function():

        for i in range(10):
            example = dataset["train"][i]
            token_ids = tokenize_function(example)

    train_ds = [tokenize_function(dataset["train"][i]) for i in range(1000)]
    test_ds = [tokenize_function(dataset["test"][i]) for i in range(1000)]
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=1, drop_last=True)  # type: ignore
    eval_dataloader = DataLoader(test_ds, batch_size=1, drop_last=True)  # type: ignore

    optimizer = SGD(phi2.model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    # progress_bar = tqdm(range(num_training_steps))
    logits = 51200

    phi2.model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            question, ans = batch
            # print(batch)
            outputs = phi2.model(question.to(device))
            outputs_reshape = outputs.logits.view(-1, outputs.logits.shape[-1])
            ans = ans.reshape((ans.shape[-1],)).to(device)
            print(question.shape, ans.shape, outputs_reshape.shape)
            loss = loss_fn(outputs_reshape, ans)

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # progress_bar.update(1)
            break

        for batch in eval_dataloader:
            question, ans = batch
            outputs = phi2.model(question.to(device))
            outputs_reshape = outputs.logits.view(-1, outputs.logits.shape[-1])
            ans = ans.reshape((ans.shape[-1],)).to(device)
            print(question.shape, ans.shape, outputs_reshape.shape)
            loss = loss_fn(outputs_reshape, ans)
            break

import pytest
@pytest.mark.skip(reason="this test takes too long")
def test_main():
    main()

if __name__ == "__main__":
    test_main()


