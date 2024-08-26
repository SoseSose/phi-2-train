# %%
from transformers import DataCollatorForLanguageModeling
import sqlite3
from pathlib import Path
import torch


from dataclasses import asdict
from data_processing.logical_op_data_module import LogicalOpDataModule
from data_processing.logical_op_ds_make import load_logical_tasks
from data_processing.arc_preprocess import ArcTask, ArcTaskSet
from torch.utils.data import DataLoader
from torch.optim import SGD
from transformers import get_scheduler
import torch
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from utils import fix_random_seed


import mlflow.pytorch
from architectures.phi2 import Phi2_light
from config.train_phi2_logi_op import DataModuleConfig, MLFLowConfig, TrainConfig
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import pytest

# %%
model = Phi2_light("D:/models/phi2", 2e-4)
model.tokenizer.pad_token = model.tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=model.tokenizer, mlm=False)
model.build()

# %%


def test_model_param_dtype():
    model = Phi2_light("D:/models/phi2", 2e-4)
    model.build()
    for param in model.parameters():
        assert param.dtype == torch.bfloat16


# %%
data_module = LogicalOpDataModule(
    train_collate_fn=model.get_tokenize_func(),
    **asdict(DataModuleConfig()),
)
data_module.prepare_data()
data_module.setup("fit")

# %%
for batch in data_module.train_dataloader():
    # print(batch["input"].shape)
    # batch = data_collator(batch["input"])
    print(batch)
    print(type(batch))
    print(batch["input"].dim())
    print(batch["input"].shape)
    batch["input"] = batch["input"][:100]
    batch["labels"] = batch["labels"][:100]
    batch["attention_mask"] = batch["attention_mask"][:100]
    with torch.autocast(device_type="cuda"):
        out = model.training_step(batch, 0)
        print(out.shape)
    break

# %%
from datasets import Dataset

tokenizer = model.tokenizer


def f(example):
    tokenized = tokenizer(example["text"])
    return tokenized


dataset = Dataset.from_dict({"text": ["あいうえおかきくけこ", "さしすせそ"]})
dataset = Dataset.from_dict({"text": ["あいうえおかきくけこ" + model.tokenizer.eos_token]})
mapped_dataset = dataset.map(f, remove_columns=["text"])
# print(mapped_dataset)
# for data in mapped_dataset:
#     print(data)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
batch = data_collator(list(mapped_dataset))
print(batch)

# %%


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_few_train_step():

    trainer = Trainer(
        max_epochs=1,  # 1エポックだけ実行
        limit_train_batches=10,  # トレーニングは10ステップだけ
        limit_val_batches=10,  # 検証も10ステップだけ
        accelerator="gpu",
        # devices=1,
        fast_dev_run=True,
        precision="bf16",
    )

    model = Phi2_light("D:/models/phi2", 2e-4)
    model.build()

    # # for module in model.modules():
    # #     size = sum(int(np.prod(p.shape)) for p in module.parameters())
    # #     print(size)
    # # print(model.parameters())

    data_module = LogicalOpDataModule(
        train_collate_fn=model.get_tokenize_func(),
        **asdict(DataModuleConfig()),
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
    )


if __name__ == "__main__":
    test_few_train_step()

# %%


def training_loop(
    train_cfg: TrainConfig,
    data_module_config: DataModuleConfig,
    mlf_cfg: MLFLowConfig,
):
    model = Phi2_light("D:/models/phi2", 2e-4)
    data_module = LogicalOpDataModule(
        train_collate_fn=model.get_tokenize_func(),
        **data_module_config.to_dict(),
    )

    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{Val_F1_Score:.2f}",
        monitor="Val_F1_Score",
        mode="max",
        verbose=True,
        save_top_k=1,
    )

    trainer = Trainer(
        callbacks=[
            EarlyStopping(
                monitor="Val_F1_Score",
                min_delta=train_cfg.min_delta,
                patience=train_cfg.patience,
                verbose=True,
                mode="max",
            ),
            checkpoint_callback,
        ],
        default_root_dir=train_cfg.model_checkpoint_dir,
        fast_dev_run=bool(train_cfg.debug_mode_sample),
        max_epochs=train_cfg.max_epochs,
        max_time=train_cfg.max_time,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
    )

    db_path = mlf_cfg.mlflow_tracking_uri
    Path(db_path).mkdir(parents=True, exist_ok=True)
    sqlite3.connect(db_path)
    mlflow.set_tracking_uri(f"sqlite:///{mlf_cfg.mlflow_tracking_uri}")
    mlflow.set_experiment(mlf_cfg.mlflow_experiment_name)

    with mlflow.start_run(
        run_name=mlf_cfg.mlflow_run_name,
        description=mlf_cfg.mlflow_description,
    ):
        trainer.fit(model=model, datamodule=data_module)
        best_model_path = checkpoint_callback.best_model_path

        trainer.test(model=model, datamodule=data_module)
        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=best_model_path,
        )

    return model, data_module


fix_random_seed()


##!くそ長い時間がかかるのでlogical_opに置き換える。
class Model:
    def __init__(self) -> None:
        device = torch.device("cuda:0")
        self.phi2 = Phi2("D:/models/phi2", device=device)
        self.max_length = int(self.phi2.MAX_TOKENS * 3 / 5)

    def tokenize_function(self, examples):
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
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
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
        def split_q_n_a(examples: ArcTask):
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


# if __name__ == "__main__":
#     test_main()
