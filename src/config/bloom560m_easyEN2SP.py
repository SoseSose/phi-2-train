import os
from dataclasses import dataclass
from pathlib import Path
import mlflow.pytorch

import sqlite3
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def get_trainer():
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{train_loss:.2f}",
        monitor="val_loss",
        mode="min",
        verbose=True,
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta= 0.005,
        patience= 10,
        verbose=True,
        mode="min",
    )

    trainer = Trainer(
        callbacks=[early_stopping, checkpoint_callback],
        default_root_dir= os.path.join(
            Path(__file__).parents[2],
            "model-checkpoints",
        ),
        max_epochs=30,
        max_time={"hours": 3},
        accelerator="gpu",
        precision="bf16-mixed",
        devices=1,
        # fast_dev_run=True,
    )
    return trainer


class MLFlowExperimentManager:
    def __init__(self):
        tracking_uri: str = "result/mlruns.db"
        experiment_name: str = "bloom560m_easyEN2SP"
        run_name: str = "op-check-run"
        description: str = """
        bloom560mで簡単なENからSPへの変換を学習する
        """

        db_path = Path(tracking_uri)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(db_path)
        mlflow.set_tracking_uri(f"sqlite:///{tracking_uri}")
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(
            run_name=run_name,
            description=description,
        )

    def __enter__(self):
        return self.run.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.run.__exit__(exc_type, exc_val, exc_tb)
