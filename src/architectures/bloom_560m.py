# %%
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from torch.optim import SGD, AdamW, Optimizer
from torchmetrics.text import Perplexity
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

MODEL_NAME = "bigscience/bloom-560m"


@dataclass
class Bloom560m_tokenizer_params:
    trust_remote_code: bool = False
    padding_side: str = "right"


def get_bloom560m_tokenizer(save_dir: str) -> AutoTokenizer:
    tokenizer_save_path = Path(save_dir) / MODEL_NAME

    if (tokenizer_save_path / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_save_path,
            **asdict(Bloom560m_tokenizer_params()),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            **asdict(Bloom560m_tokenizer_params()),
        )
        tokenizer.save_pretrained(tokenizer_save_path)

    return tokenizer



@dataclass
class Bloom560m_params:
    device_map: str = "cuda"
    trust_remote_code: bool = True
    torch_dtype: torch.dtype = torch.float16
    use_cache: bool = True
    # use_flash_attention_2:bool=True


class Bloom560m(LightningModule):
    def __init__(
        self,
        save_dir: str,
        lr: float,
    ):
        super().__init__()
        self.lr = lr
        self.tokenizer = get_bloom560m_tokenizer(save_dir)
        self.save_dir = save_dir

    # def build_model(self):
        model_save_path = Path(self.save_dir) / MODEL_NAME
        if (model_save_path / "config.json").exists():
            self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                model_save_path,
                **asdict(Bloom560m_params()),
            )

        else:
            self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                **asdict(Bloom560m_params()),
            )
            self.model.save_pretrained(model_save_path)

    def forward(self, batch):
        return self.model.forward(**batch)

    def training_step(self, batch, batch_idx):
        model_answer_logits = self.model.forward(**batch)
        self.log(
            "train_loss",
            model_answer_logits.loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return model_answer_logits.loss

    def validation_step(self, batch, batch_idx):
        model_answer_logits = self.model.forward(**batch)
        self.log("val_loss", model_answer_logits.loss)

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        model_answer_logits = self.model.forward(**batch)
        self.log("test_loss", model_answer_logits.loss)

    def predict_step(self, batch, batch_idx):
        # print(batch)
        print(batch["input_ids"].shape)
        label = batch["input_ids"][:, -2]
        question = batch["input_ids"][:, :-2]
        question_str = self.tokenizer.batch_decode(question)
        label_str = self.tokenizer.batch_decode(label)

        output = self.model.generate(question, max_length=100)
        gen_text = self.tokenizer.batch_decode(output)
        return {"question":question_str, "gen_text": gen_text, "label": label_str}
        # return {"original_text": original_text}

    def configure_optimizers(self) -> Optimizer:
        # oprimizer = AdamW(
        #     params=self.model.parameters(),
        #     lr=self.lr,
        #     weight_decay=0.03,
        # )
        oprimizer = SGD(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=0.03,
        )

        schedulers = get_linear_schedule_with_warmup(
            optimizer=oprimizer, num_warmup_steps=10, num_training_steps=1000
        )
        return [oprimizer], [schedulers]
