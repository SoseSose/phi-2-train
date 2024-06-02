from pathlib import Path
import mlflow
import sqlite3
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

def get_trainer():
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.03,
        patience=10,
        verbose=True,
        mode="min",
    )

    trainer = Trainer(
        callbacks=[early_stopping],
        logger=False,  # ログにはMLflowを使用するためFalse
        enable_checkpointing=False,  # チェックポイントにはMLflowを使用するためFalse
        max_epochs=30,
        max_time={"hours": 3},
        accelerator="gpu",
        precision="bf16-mixed",
        devices=1,
    )
    return trainer

class MLFlowExperimentManager:
    TRACKING_SQL_PATH = "result/mlruns.db"
    EXPERIMENT_NAME = "bloom560m_easyEN2SP"
    RUN_NAME = "op-check-run"
    DESCRIPTION = """
    bloom560mで簡単なENからSPへの変換を学習する
    """

    def __init__(self):
        self._setup_tracking_sql()
        self._setup_experiment()
        self.run = self._start_run()

    def _setup_tracking_sql(self):
        db_path = Path(self.TRACKING_SQL_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(db_path)
        mlflow.set_tracking_uri(f"sqlite:///{self.TRACKING_SQL_PATH}")

    def _setup_experiment(self):
        mlflow.enable_system_metrics_logging()
        mlflow.set_experiment(self.EXPERIMENT_NAME)

    def _start_run(self):
        mlflow.pytorch.autolog(
            log_models=False,  #Trueにすると最後の状態のモデルが保存される.今回は最後の状態ではなく,metricが良い状態のcheckpointを使用するためFalse
            checkpoint=True,
            checkpoint_metric="val_loss",
            checkpoint_mode="min",
        )
        return mlflow.start_run(
            run_name=self.RUN_NAME,
            description=self.DESCRIPTION,
        )

    def __enter__(self):
        return self.run.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.run.__exit__(exc_type, exc_val, exc_tb)

#工夫した点,Mlflowの依存をMLFlowExperimentMangerのみにした.パラメータを設定する部分はコンストラクタに直接代入して,型を間違えないように.mlflow.pytorch.autologを活用,チェックポイント生成など自動で行ってくれる.(lightningのcallbackでもcheckpoint作成はできるが,lightningで作ったあと,mlflowのartifactに登録しないといけないので,依存関係がゴチャゴチャすることを避ける.もっと詳細にカスタマイズしたいならlightningのcallbackを使用するのもあり.adamだとlossが発散してうまく学習しなかった.