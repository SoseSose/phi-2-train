# %%
import torch
from lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms
from pathlib import Path

from data_processing.logical_op_ds_make import load_logical_tasks, save_logical_tasks


class LogicalOpDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_num: int,
        eval_num: int,
        batch_size: int,
        eval_ratio: float,
    ):
        super().__init__()

        self.train_dir = Path(data_dir) / "train"
        self.train_num = train_num
        self.eval_num = eval_num
        self.eval_dir = Path(data_dir) / "eval"
        self.batch_size = batch_size

        if not 0.0 < eval_ratio < 1.0:
            raise ValueError("split_ratio has to be between 0.0 and 1.0")

        self.eval_ratio = eval_ratio

    def prepare_data(self):
        save_logical_tasks(self.train_dir, self.train_num, 10)
        save_logical_tasks(self.eval_dir, self.eval_num, 10)

    def setup(self, stage: str):
        if stage == "fit":
            ds = load_logical_tasks(self.train_dir)
            eval_num = len(ds) * self.eval_ratio
            print(eval_num, len(ds) - eval_num)
            self.train_ds, self.eval_ds = random_split(
                ds,
                [int(len(ds) - eval_num), int(eval_num)],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test" and stage == "predict":
            self.eval_ds = load_logical_tasks(self.train_dir)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.eval_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.eval_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.eval_ds, batch_size=self.batch_size)


import pytest

class TestLogicalOpDataModule:
    
    def test_setup(self, tmp_path: Path):

        data_module = LogicalOpDataModule(
            data_dir=tmp_path / "logical_op",
            train_num=100,
            eval_num=20,
            batch_size=32,
            eval_ratio=0.2,
        )
        data_module.prepare_data()

        assert (data_module.train_dir).exists()
        assert (data_module.eval_dir).exists()

        data_module.setup("fit")
        assert len(data_module.train_ds) == 80
        assert len(data_module.eval_ds) == 20

        data_module.setup("test")
        assert len(data_module.eval_ds) == 20

        data_module.setup("predict")
        assert len(data_module.eval_ds) == 20
