#%%
from pathlib import Path
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
from torch.optim import Optimizer, AdamW
from torchmetrics import Accuracy
import random

from transformers.modeling_outputs import CausalLMOutputWithPast

from lightning import LightningModule


from dataclasses import dataclass,asdict

@dataclass
class Phi2_params:

    device_map:str="cuda"
    trust_remote_code:bool=True
    # torch_dtype:torch.dtype=torch.float16 
#%%

class Phi2_light(LightningModule):
    def __init__(
        self,
        save_dir: str,
        lr: float,
    ):
        super().__init__()
        self.save_dir = save_dir

        self.__is_instanced_model = False
        self.lr = lr

        if Path(self.save_dir).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.save_dir,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/phi-2"
            )  # download phi2
            self.tokenizer.save_pretrained(self.save_dir)  # save phi2

    def build(self):
        if not self.__is_instanced_model:
            self.__is_instanced_model = True
            if not Path(self.save_dir).exists():
                # download phi2
                self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                    "microsoft/phi-2",
                    **asdict(Phi2_params()),
                )

                # save phi2
                self.model.save_pretrained(self.save_dir)

            else:
                self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                    self.save_dir,
                    **asdict(Phi2_params()),
                )

    def get_tokenize_func(self):
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
                return self.tokenizer.encode(
                    text,
                    truncation=True,
                )

            tokenized_que = tokenize(question)
            tokenized_ans = tokenize(answer + self.tokenizer.eos_token)
            # 元のデータにはeos_tokenがないので,ここで追加する

            random_point = random.randint(0, len(tokenized_ans))
            # random_pointは最後のeos_tokenも含む可能性はあることは確認ずみ
            clipped_tokenized_ans = tokenized_ans[:random_point]

            tokenized_text = tokenized_que + clipped_tokenized_ans

            def truncate(text):
                max_len = self.tokenizer.model_max_length
                if len(text) > max_len:
                    return text[:max_len]
                else:
                    return text

            tokenized_text = truncate(tokenized_text)
            labels = tokenized_text.copy()
            input_text = tokenized_text.copy()
            attention_mask = [1] * len(input_text)
            input_text[-1] = self.tokenizer.eos_token_id
            # 1トークン予測するのでinputの最終トークンはeos_tokenにする

            # attention_mask[-1] = 0

            def to_val_tensor(val):
                val = torch.tensor(val)
                if val.dim() == 1:
                    val = val.unsqueeze(0)
                return val
            
            input_text = to_val_tensor(input_text).to(torch.long)
            attention_mask = to_val_tensor(attention_mask).to(self.model.dtype)
            labels = to_val_tensor(labels).to(dtype=torch.long)

            return {
                "input": input_text,
                "labels": labels,
                "attention_mask": attention_mask,
            }

        return tokenize_func
    


    def training_step(self, batch, batch_idx):
        # self.build()

        model_answer_logits = self.model.forward(
            batch["input"].to(self.model.device),
            attention_mask=batch["input"].to(self.model.device),
            labels=batch["labels"].to(self.model.device),
        )

        return model_answer_logits.loss
        # self.log_dict(metrics, on_epoch=True, on_step=False)

    def _encode(self, question: str) -> CausalLMOutputWithPast:
        token_ids = self.tokenizer.encode(
            question,
            truncation=True,
            # モデルのシーケンス最大長で生成は打ち切られるので,この処理をしていれば,このあとのgenerateでエラーが出ることはない
            return_tensors="pt",
            return_length=True,
        )
        return token_ids

    def generate_ids(self, question: str) -> str:
        token_ids = self._encode(question)
        question_token_num = len(token_ids[0])
        model_max_token_num = self.model.config.max_position_embeddings
        answer_max_length = min(
            question_token_num + question_token_num * 1 / 5,
            model_max_token_num,
        )
        # 回答はふつう入力の5/1以下になると考えられるのでこの処理を
        # モデルの生成時間を減らしたいためにこの処理を追加するが
        # 生成にかかる時間は入力が1000トークンくらいで10秒くらい
        # 入力が2048トークンだと1秒くらい
        # 入力が10トークンくらいだと0.2秒
        # 無理にモデルの生成長さを制限しても,生成時間を減らすのは難しそう

        self.build()
        full_output_ids = self.model.generate(
            token_ids.to(self.model.device),
            pad_token_id=self.tokenizer.eos_token_id,
            # ないとThe attention mask ~という警告が出る。
            temperature=0.2,
            do_sample=True,
            max_length=answer_max_length,
        )
        # return full_output_ids

        ouput_ids_delete_question = full_output_ids[0][question_token_num:]
        return ouput_ids_delete_question

    def get_ans(self, question: str) -> str:
        output_ids = self.generate_ids(question)
        # print(output_ids)
        # return output_ids
        answer = self.tokenizer.decode(output_ids)
        return answer

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


import pytest

@pytest.mark.skip(reason="this is slow")
class TestPhi2_pl:
    def __init__(self):
        self.phi2 = Phi2_light(
            "D:/models/phi2", 2e-4, "auto", Accuracy(task="multiclass", num_classes=10)
        )

    def test_get_ans(self):
        answer = self.phi2.get_ans("What is the sum of 1 and 2?")
        assert answer is not None

    def test_training_step(self):
        question = "What is the sum of 1 and 2?"
        collect_answer = "3"
        loss = self.phi2.training_step((question, collect_answer), 0)
        assert loss is not None

    def test_tokenized_func(self):
        question = "What is the sum of 1 and 2?"
        ans = "3"
        tokenized = self.phi2.get_tokenize_func()(question, ans)
        assert tokenized["input"][-1] == self.phi2.tokenizer.eos_token_id
        assert tokenized["input"][:-1] == tokenized["labels"][:-1]
        assert tokenized["attention_mask"][-1] == 0
        assert 0 not in tokenized["attention_mask"][:-1]
    
    def test_generate_ids(self):
        question = "What is the sum of 1 and 2?"
        output_ids = self.phi2.generate_ids(question)
        assert output_ids is not None
