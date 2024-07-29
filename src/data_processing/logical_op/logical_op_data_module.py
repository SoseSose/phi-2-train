# %%
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

from data_processing.logical_op_ds_make import load_logical_tasks, save_logical_tasks


class LogicalOpDs(Dataset):
    def __init__(self, path:Path):
        self.tasks = load_logical_tasks(path)

    def __getitem__(self, index:int):
        return self.tasks[index]

    def __len__(self):
        return len(self.tasks)


class LogicalOpDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_num: int,
        eval_num: int,
        task_len:int,
        batch_size: int,
        train_collate_fn=None,
    ):
        super().__init__()

        self.train_dir = Path(data_dir) / "train"
        self.train_num = train_num
        self.eval_num = eval_num
        self.task_len = task_len
        self.eval_dir = Path(data_dir) / "eval"
        self.batch_size = batch_size
        self.train_collate_fn = train_collate_fn



    def prepare_data(self):
        save_logical_tasks(self.train_dir, self.train_num, self.task_len)
        save_logical_tasks(self.eval_dir, self.eval_num, self.task_len)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = LogicalOpDs(self.train_dir)
            self.eval_ds = LogicalOpDs(self.eval_dir)

        if stage == "test" and stage == "predict":
            self.eval_ds = LogicalOpDs(self.eval_dir)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=self.train_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.eval_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.eval_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.eval_ds, batch_size=self.batch_size)
    


import pytest

class TestLogicalOpDataModule:
    
    def test_setup(self, tmp_path: Path):

        train_num = 100
        eval_num = 20
        data_module = LogicalOpDataModule(
            data_dir=tmp_path / "logical_op",
            train_num=train_num,
            eval_num=eval_num,
            task_len=5,
            batch_size=1,
        )
        data_module.prepare_data()

        assert (data_module.train_dir).exists()
        assert (data_module.eval_dir).exists()

        data_module.setup("fit")
        assert len(data_module.train_ds) == 100
        assert len(data_module.eval_ds) == 20

        data_module.setup("test")
        assert len(data_module.eval_ds) == 20

        data_module.setup("predict")
        assert len(data_module.eval_ds) == 20
        

    def test_train_dataloader(self, tmp_path: Path):
        data_module = LogicalOpDataModule(
            data_dir=tmp_path / "logical_op",
            train_num=100,
            eval_num=20,
            task_len = 5,
            batch_size=1,
        )
        data_module.prepare_data()
        data_module.setup("fit")
        dl = data_module.train_dataloader()
        assert len(dl) == 3
        # for batch in dl:
        #     print(batch["input"])
        #     print(batch["output"])
            # break
