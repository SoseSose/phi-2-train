import sqlite3
from pathlib import Path

import mlflow.pytorch
import torch
from architectures.phi2 import Phi2_light
from config.train_config import DataModuleConfig, MLFLowConfig, TrainConfig
from data_processing.logical_op_data_module import LogicalOpDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def training_loop(
    train_cfg: TrainConfig,
    data_module_config: DataModuleConfig,
    mlf_cfg: MLFLowConfig,
):
    model = Phi2_light("D:/models/phi2", 2e-4)
    data_module = LogicalOpDataModule(
        train_collate_fn=model.get_tokenize_func(),
        **data_module_config.asdict(),
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
