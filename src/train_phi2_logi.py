#%%
from transformers import DataCollatorForLanguageModeling
import sqlite3
from pathlib import Path
import torch


from dataclasses import asdict
import mlflow.pytorch
from architectures.phi2 import Phi2_light
from config.train_phi2_logi_op import DataModuleConfig, MLFLowConfig, TrainConfig
from data_processing.logical_op_data_module import LogicalOpDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import pytest
#%%
model = Phi2_light("D:/models/phi2", 2e-4)
model.tokenizer.pad_token = model.tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=model.tokenizer, mlm=False)
model.build()

#%%

def test_model_param_dtype():
    model = Phi2_light("D:/models/phi2", 2e-4)
    model.build()
    for param in model.parameters():
        assert param.dtype == torch.bfloat16

#%%
data_module = LogicalOpDataModule(
    train_collate_fn=model.get_tokenize_func(),
    **asdict(DataModuleConfig()),
)
data_module.prepare_data()
data_module.setup("fit")

#%%
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

#%%
from datasets import Dataset
tokenizer = model.tokenizer
def f(example):
    tokenized = tokenizer(example['text'])
    return tokenized

dataset = Dataset.from_dict({"text": ["あいうえおかきくけこ", "さしすせそ"]})
dataset = Dataset.from_dict({"text": ["あいうえおかきくけこ"+model.tokenizer.eos_token]})
mapped_dataset = dataset.map(f, remove_columns=['text'])
# print(mapped_dataset)
# for data in mapped_dataset:
#     print(data)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
batch = data_collator(list(mapped_dataset))
print(batch)

#%%

@pytest.mark.filterwarnings('ignore::RuntimeWarning')
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

#%%

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
