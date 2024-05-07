import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    pretrained_model: str = "roberta-base"
    num_classes: int = 2
    lr: float = 2e-4
    device = "cuda:0"

    max_length: int = 128
    batch_size: int = 256
    num_workers: int = os.cpu_count()
    max_epochs: int = 10
    max_time: dict[str, float] = field(default_factory=lambda: {"hours": 3})
    model_checkpoint_dir: str = os.path.join(
        Path(__file__).parents[2],
        "model-checkpoints",
    )
    min_delta: float = 0.005
    patience: int = 10




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
