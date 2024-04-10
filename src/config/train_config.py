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
    debug_mode_sample: int = 100 
    max_time: dict[str, float] = field(default_factory=lambda: {"hours": 3})
    model_checkpoint_dir: str = os.path.join(
        Path(__file__).parents[2],
        "model-checkpoints",
    )
    min_delta: float = 0.005
    patience: int = 4

@dataclass
class DataModuleConfig:
    data_dir: str = "data/logical_op"
    train_num: int = 1000
    eval_num: int = 200
    task_len: int = 5
    batch_size: int = 1

@dataclass
class MLFLowConfig: 

    # MLflow
    mlflow_tracking_uri: str = "result/mlruns.db"
    mlflow_artifact_location: str = "result/artifacts"
    mlflow_experiment_name: str = "phi2-train-logical-op"
    mlflow_run_name: str = "first-run"
    mlflow_description: str =  """
    オリジナルのPHI2から論理演算の学習をさせる.
    学習では自作の論理演算データセットでtest_input+test_outputの途中までを入力として,その次の1トークンを出力として学習させる.
    evalではArc内にある論理演算タイプのタスクを解かせて,その正解数が一つでも増えれば,そのモデルを保存する.
    その他,自作データセットのロスが減っているか,通常の文章生成データセットの正解率が減っていないかもチェックする.
    
    """