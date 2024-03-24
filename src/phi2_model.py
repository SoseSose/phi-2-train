# %%
from pathlib import Path
from re import A
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from abc import abstractclassmethod, ABC
from torch.optim import Optimizer, AdamW
import torch.nn.functional as F
from torchmetrics import Accuracy

from lightning import LightningModule


class BaseModel(ABC):
    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def build(self):
        pass

    @abstractclassmethod
    def get_token_num_and_answer(self, question: str) -> tuple[str, int]:
        pass


class Phi2_light(LightningModule):
    MAX_TOKENS = 2048

    def __init__(
        self,
        save_dir: str,
        lr: float,
        acc_fn:Accuracy,
        device: str = "auto",
        
    ):
        self.save_dir = save_dir
        self.device = device
        self.__isnt_instanced = True
        self.acc_fn = acc_fn
        self.lr

    def build(self):
        if self.__isnt_instanced:
            self.__isnt_instanced = False
            if not Path(self.save_dir).exists():
                # download phi2
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
                self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
                # save phi2
                self.tokenizer.save_pretrained(self.save_dir)
                self.model.save_pretrained(self.save_dir)

            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.save_dir,
                    trust_remote_code=True,
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.save_dir,
                    torch_dtype="auto",
                    device_map=self.device,
                    trust_remote_code=True,
                )

    def forward(self, question: str)->str:
        self.build()
        token_ids = self.tokenizer.encode(
            question,
            add_special_tokens=True,
            # add_special_tokens=False,
        )
        token_ids = torch.tensor(token_ids).unsqueeze(0)
        token_num = len(token_ids[0])
        if token_num > self.MAX_TOKENS:
            answer = "token_num > {}".format(self.MAX_TOKENS)
        else:
            output_ids = self.generate(token_ids)
            answer = self.tokenizer.decode(output_ids[0][token_ids.size(1) :])

        self.tokenizer.decode(token_ids)
        output_ids = self.model.generate(
            token_ids.to(self.model.device),
            pad_token_id=self.tokenizer.eos_token_id,
            # ないとThe attention mask ~という警告が出る。
            temperature=0.2,
            do_sample=True,
            max_length=self.MAX_TOKENS,
        )
        return answer


    def training_step(self, batch, batch_idx):
        question, answer = batch
        model_answer = self.forward(batch)
        loss = F.mse_loss(model_answer, answer)

        # self.log_dict(metrics, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        question, answer = batch
        model_answer = self.forward(batch)
        self.acc_fn(model_answer, answer)
        # return metrics

    def configure_optimizers(self) -> Optimizer:
        return AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=0.0,
        )


class Phi2(BaseModel):
    MAX_TOKENS = 2048

    def __init__(
        self,
        save_dir: str,
        device: str = "auto",
    ):
        self.save_dir = save_dir
        self.device = device
        self.__isnt_instanced = True

    def build(self):
        if self.__isnt_instanced:
            self.__isnt_instanced = False
            if not Path(self.save_dir).exists():
                # download phi2
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
                self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
                # save phi2
                self.tokenizer.save_pretrained(self.save_dir)
                self.model.save_pretrained(self.save_dir)

            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.save_dir,
                    trust_remote_code=True,
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.save_dir,
                    torch_dtype="auto",
                    device_map=self.device,
                    trust_remote_code=True,
                )

    def generate(self, token_ids):
        self.build()
        output_ids = self.model.generate(
            token_ids.to(self.model.device),
            pad_token_id=self.tokenizer.eos_token_id,
            # ないとThe attention mask ~という警告が出る。
            temperature=0.2,
            do_sample=True,
            max_length=self.MAX_TOKENS,
        )
        return output_ids

    def get_token_num_and_answer(self, question: str) -> tuple[str, int]:
        self.build()
        try:
            with torch.no_grad():
                token_ids = self.tokenizer.encode(
                    question,
                    add_special_tokens=True,
                    # add_special_tokens=False,
                )
                token_ids = torch.tensor(token_ids).unsqueeze(0)
                token_num = len(token_ids[0])
                if token_num > self.MAX_TOKENS:
                    answer = "token_num > {}".format(self.MAX_TOKENS)
                else:
                    output_ids = self.generate(token_ids)
                    answer = self.tokenizer.decode(output_ids[0][token_ids.size(1) :])

        except Exception as e:
            print("error")
            print(e)
            answer = token_num = "error"

        return answer, token_num


import pytest


@pytest.mark.skip(reason="this is slow")
def test_get_token_num_and_anser():
    question = "What is the sum of 1 and 2?"
    phi2 = Phi2("D:/models/phi2")

    answer, token_num = phi2.get_token_num_and_answer(question)

    print(answer)
    assert answer is not None
    assert token_num is not None


if __name__ == "__main__":
    test_get_token_num_and_anser()
