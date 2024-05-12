# %%
import sqlite3
from pathlib import Path

import mlflow.pytorch
import utils
from architectures.bloom_560m import Bloom560m
from config.train_bloom560m_easyEN2SP import MLFLowConfig,get_trainer
from data_processing.easy_ds_EN_to_SP import EasyEnToSpDM


utils.ini_setting()


def training_loop(
    mlf_cfg: MLFLowConfig,
):
    model = Bloom560m("D:/models", 0.001)

    easy_en_to_sp_dm = EasyEnToSpDM(model.tokenizer, 1)

    trainer = get_trainer()

    db_path = Path(mlf_cfg.mlflow_tracking_uri)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(db_path)
    mlflow.set_tracking_uri(f"sqlite:///{mlf_cfg.mlflow_tracking_uri}")
    mlflow.set_experiment(mlf_cfg.mlflow_experiment_name)

    with mlflow.start_run(
        run_name=mlf_cfg.mlflow_run_name,
        description=mlf_cfg.mlflow_description,
    ) as run:
        run.set_tag("test tag")
        trainer.fit(model=model, datamodule=easy_en_to_sp_dm)
        trainer.test(model=model, datamodule=easy_en_to_sp_dm)
        predictions = trainer.predict(
            model=model,
            datamodule=easy_en_to_sp_dm,
            return_predictions=True,
        )
    for pred in predictions:
        print(pred)

    return model


if __name__ == "__main__":
    training_loop(mlf_cfg=MLFLowConfig())

