# %%
import sqlite3
from pathlib import Path

import mlflow.pytorch
import utils
from architectures.bloom_560m import Bloom560m, get_bloom560m_tokenizer
from config.train_bloom560m_easyEN2SP import MLFLowConfig, TrainConfig
from data_processing.easy_ds_EN_to_SP import (
    EasyEnToSpDM,
    get_masked_ds,
    predict_training_set,
    try_print_iterative_gen,
)
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

utils.ini_setting()


def training_loop(
    train_cfg: TrainConfig,
    mlf_cfg: MLFLowConfig,
):
    tokenizer = get_bloom560m_tokenizer("D:/models")
    easy_en_to_sp_dm = EasyEnToSpDM(tokenizer, 1)
    model = Bloom560m("D:/models", 0.001)
    predict_training_set(model.model, tokenizer)
    try_print_iterative_gen(model.model, tokenizer)

    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{train_loss:.2f}",
        monitor="val_loss",
        mode="min",
        verbose=True,
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=train_cfg.min_delta,
        patience=train_cfg.patience,
        verbose=True,
        mode="min",
    )

    trainer = Trainer(
        callbacks=[early_stopping, checkpoint_callback],
        default_root_dir=train_cfg.model_checkpoint_dir,
        max_epochs=train_cfg.max_epochs,
        max_time=train_cfg.max_time,
        accelerator="gpu",
        precision="bf16-mixed",
        devices=1,
        # fast_dev_run=True,
    )

    db_path = Path(mlf_cfg.mlflow_tracking_uri)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(db_path)
    mlflow.set_tracking_uri(f"sqlite:///{mlf_cfg.mlflow_tracking_uri}")
    mlflow.set_experiment(mlf_cfg.mlflow_experiment_name)

    with mlflow.start_run(
        run_name=mlf_cfg.mlflow_run_name,
        description=mlf_cfg.mlflow_description,
    ):
        trainer.fit(model=model, datamodule=easy_en_to_sp_dm)
        best_model_path = checkpoint_callback.best_model_path

        # trainer.test(model=model, datamodule=easy_en_to_sp_dm)

    return model, best_model_path


if __name__ == "__main__":
    training_loop(train_cfg=TrainConfig(), mlf_cfg=MLFLowConfig())


# %%
def test_get_masked_ds():
    # sanity check that our format is correct
    # we'd expect -100 for the human text and the actual token(s) for the assistant text
    tokenizer = get_bloom560m_tokenizer("D:/models")
    masked_dataset = get_masked_ds(tokenizer)
    label_ex = masked_dataset[0]["labels"]
    print(f"{label_ex=}")
    # let's see just the non-masked text
    non_masked_text = tokenizer.decode(
        label_ex[label_ex != -100], skip_special_tokens=False
    )
    assert non_masked_text == " perro</s>"
    print(f"non masked text: {non_masked_text}")
    # let's see just the masked text
    # -100 is not a real token, convert to something the tokenizer understands
    label_ex[label_ex == -100] = 0
    full_lable = tokenizer.decode(label_ex, skip_special_tokens=False)
    print(f"full 'label': {full_lable}")
    assert (
        full_lable
        == "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> perro</s>"
    )
