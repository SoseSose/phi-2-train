import os
from dataclasses import dataclass, field
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

@dataclass
class TrainConfig:
    pretrained_model: str = "roberta-base"
    num_classes: int = 2
    lr: float = 2e-4
    device = "cuda:0"

    max_length: int = 128
    batch_size: int = 256

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
        max_epochs=10,
        max_time={"hours": 3},
        accelerator="gpu",
        precision="bf16-mixed",
        devices=1,
        # fast_dev_run=True,
    )
    return trainer

@dataclass
class MLFLowConfig:
    # MLflow
    mlflow_tracking_uri: str = "result/mlruns.db"
    mlflow_artifact_location: str = "result/artifacts"
    mlflow_experiment_name: str = "bloom560m_easyEN2SP"
    mlflow_run_name: str = "op-check-run"
    mlflow_description: str = """
    bloom560mで簡単なENからSPへの変換を学習する
    """