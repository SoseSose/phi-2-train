# %%
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from torch.optim import AdamW, Optimizer
from torchmetrics import Accuracy
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from transformers.modeling_outputs import CausalLMOutputWithPast

MODEL_NAME = "bigscience/bloom-560m"


@dataclass
class Bloom560m_tokenizer_params:
    trust_remote_code: bool = False
    padding_side: str = "right"


def get_bloom560m_tokenizer(save_dir: str):
    tokenizer_save_path = Path(save_dir) / MODEL_NAME

    if (tokenizer_save_path/"tokenizer.json").exists():
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
        tokenizer = tokenizer

    return tokenizer


def get_tokenize_func(tokenizer):
    def tokenize_func(inputs):
        """
        questionとanswerをトークン化して,結合し["input"]は最終トークンがeos_token_idになるよう,["labels"]は元のデータになるように,attention_maskも最終トークンのみ0で他は1になるようにする
        !!バッチには対応していない!!
        """
        inputs = inputs[0]
        # なぜかcollate_fnを使うとinputsがリストになるので,ここで取り出す
        question = inputs["input"]
        answer = inputs["output"]

        def tokenize(text):
            return tokenizer.encode(
                text,
                truncation=True,
            )

        tokenized_que = tokenize(question)
        tokenized_ans = tokenize(answer + tokenizer.eos_token)
        # 元のデータにはeos_tokenがないので,ここで追加する

        random_point = random.randint(0, len(tokenized_ans))
        # random_pointは最後のeos_tokenも含む可能性はあることは確認ずみ
        clipped_tokenized_ans = tokenized_ans[:random_point]

        tokenized_text = tokenized_que + clipped_tokenized_ans

        def truncate(text):
            max_len = tokenizer.model_max_length
            if len(text) > max_len:
                return text[:max_len]
            else:
                return text

        tokenized_text = truncate(tokenized_text)
        labels = tokenized_text.copy()
        input_text = tokenized_text.copy()
        attention_mask = [1] * len(input_text)
        input_text[-1] = tokenizer.eos_token_id
        # 1トークン予測するのでinputの最終トークンはeos_tokenにする

        # attention_mask[-1] = 0

        def to_val_tensor(val):
            val = torch.tensor(val)
            if val.dim() == 1:
                val = val.unsqueeze(0)
            return val

        # input_text = to_val_tensor(input_text).to(torch.long)
        # attention_mask = to_val_tensor(attention_mask).to(model.dtype)
        # labels = to_val_tensor(labels).to(dtype=torch.long)

        return {
            "input": input_text,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return tokenize_func

@dataclass
class Bloom560m_params:
    device_map: str = "cuda"
    trust_remote_code: bool = True
    torch_dtype:torch.dtype=torch.float16
    use_cache:bool=True
    # use_flash_attention_2:bool=True


class Bloom560m(LightningModule):
    def __init__(
        self,
        save_dir: str,
        lr: float,
    ):
        super().__init__()

        self.lr = lr

        model_save_path = Path(save_dir) / MODEL_NAME
        if (model_save_path/"config.json").exists():
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

        return model_answer_logits.loss
        # self.log_dict(metrics, on_epoch=True, on_step=False)


    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        question, answer = batch
        model_answer = self.get_ans(question)
        model_answer = model_answer.split("\n\n")[0]
        return {"correct": model_answer == answer}

    def configure_optimizers(self) -> Optimizer:
        return AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=0.0,
        )
