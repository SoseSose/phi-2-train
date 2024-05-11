# %%
import sqlite3
from pathlib import Path

import mlflow.pytorch
import utils
from architectures.bloom_560m import Bloom560m, get_bloom560m_tokenizer
from config.train_bloom560m_easyEN2SP import MLFLowConfig, TrainConfig, get_trainer
from data_processing.easy_ds_EN_to_SP import (
    EasyEnToSpDM,
    get_masked_ds,
    predict_training_set,
    try_print_iterative_gen,
)

utils.ini_setting()


def training_loop(
    mlf_cfg: MLFLowConfig,
):
    model = Bloom560m("D:/models", 0.001)

    easy_en_to_sp_dm = EasyEnToSpDM(model.tokenizer, 1)

    predict_training_set(model.model, model.tokenizer)
    try_print_iterative_gen(model.model, model.tokenizer)

    trainer = get_trainer()

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
        trainer.test(model=model, datamodule=easy_en_to_sp_dm)
        predictions = trainer.predict(
            model=model,
            datamodule=easy_en_to_sp_dm,
            return_predictions=True,
        )
    print(predictions)


    # predict_training_set(model.model, tokenizer)
    # try_print_iterative_gen(model.model, tokenizer)

    return model


if __name__ == "__main__":
    training_loop(mlf_cfg=MLFLowConfig())


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
