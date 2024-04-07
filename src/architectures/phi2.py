# %%
from pathlib import Path
import token
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Optimizer, AdamW
import torch.nn.functional as F
from torchmetrics import Accuracy

# from transformers.
from transformers.modeling_outputs import CausalLMOutputWithPast

from lightning import LightningModule


class Phi2_light(LightningModule):
    def __init__(
        self,
        save_dir: str,
        lr: float,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.__isnt_instanced = True
        self.lr = lr

    def build(self):
        if self.__isnt_instanced:
            self.__isnt_instanced = False
            if not Path(self.save_dir).exists():
                # download phi2
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
                self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                    "microsoft/phi-2"
                )

                # save phi2
                self.tokenizer.save_pretrained(self.save_dir)
                self.model.save_pretrained(self.save_dir)

            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.save_dir,
                    trust_remote_code=True,
                )
                self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                    self.save_dir,
                    torch_dtype="auto",
                    trust_remote_code=True,
                )

    def _encode(self, question: str) -> CausalLMOutputWithPast:
        token_ids = self.tokenizer.encode(
            question,
            truncation=True,
            # モデルのシーケンス最大長で生成は打ち切られるので,この処理をしていれば,このあとのgenerateでエラーが出ることはない
            return_tensors="pt",
            return_length=True,
        )
        return token_ids

    def forward(self, question: str) -> torch.Tensor:
        self.build()
        token_ids = self._encode(question)
        token_ids = torch.tensor(
            token_ids,
            device=self.model.device,
        )

        return self.model.forward(token_ids.to(self.model.device))

    def generate_ids(self, question: str) -> str:
        self.build()
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

        full_output_ids = self.model.generate(
            token_ids.to(self.model.device),
            pad_token_id=self.tokenizer.eos_token_id,
            # ないとThe attention mask ~という警告が出る。
            temperature=0.2,
            do_sample=True,
            max_length=answer_max_length,
        )
        ouput_ids_delete_question = full_output_ids[0][question_token_num:]

        return ouput_ids_delete_question

    def get_ans(self, question: str) -> str:
        output_ids = self.generate_ids(question)
        answer = self.tokenizer.decode(output_ids)
        return answer

    def _encode_ques_and_ans(self, question: str, ans: str) -> CausalLMOutputWithPast:
        # self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_prefix_space = True
        question_and_ans = question + ans
        tokenized_ques_and_ans = self.tokenizer(
            text=[[question], [question_and_ans]],
            padding="longest",
            # padding="max_length",
            # max_length=self.model.config.max_position_embeddings,
            return_tensors="pt",
            is_split_into_words=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_length=True,
        )
        return tokenized_ques_and_ans

    def calc_loss(self, question: str, collect_answer: str):
        tokenized_ques_and_ans = self._encode_ques_and_ans(question, collect_answer)
        padded_question_ids = (
            tokenized_ques_and_ans.input_ids[0].unsqueeze(0).to(self.model.device)
        )
        print(padded_question_ids)
        ans_ids = tokenized_ques_and_ans.input_ids[1].unsqueeze(0).to(self.model.device)
        print(ans_ids)
        attn_mask = (
            tokenized_ques_and_ans.attention_mask[0].unsqueeze(0).to(self.model.device)
        )
        print(attn_mask)

        question_mask = (
            torch.where(tokenized_ques_and_ans.attention_mask[0] == 0, 1, 0)
            .unsqueeze(0)
            .to(self.model.device)
        )

        model_answer_logits = self.model.forward(
            padded_question_ids,
            # attention_mask=attn_mask,
            # attention_mask=question_mask,
        )

        return model_answer_logits

    def training_step(self, batch, batch_idx):
        question, collect_answer = batch

        closs_entropy = torch.nn.CrossEntropyLoss()
        loss = closs_entropy(model_answer_logits, ans_ids)

        return loss
        # self.log_dict(metrics, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        question, answer = batch
        model_answer = self.forward(batch)
        # self.acc_fn(model_answer, answer)
        # return metrics

    def configure_optimizers(self) -> Optimizer:
        return AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=0.0,
        )


import pytest


@pytest.mark.skip(reason="this is slow")
class TestPhi2_pl:
    def __init__(self):
        self.model = Phi2_light(
            "D:/models/phi2", 2e-4, "auto", Accuracy(task="multiclass", num_classes=10)
        )

    def test_get_ans(self):
        answer = self.model.get_ans("What is the sum of 1 and 2?")
        assert answer is not None

    def test_training_step(self):
        question = "What is the sum of 1 and 2?"
        collect_answer = "3"
        loss = self.model.training_step((question, collect_answer), 0)
        assert loss is not None


import pytest

# %%
phi2 = Phi2_light("D:/models/phi2", 2e-4)
phi2.build()
# %%
question = "100 + 10 = "
collect_answer = "aaaa"
# bos_token = phi2.tokenizer.bos_token
# eos_token = phi2.tokenizer.eos_token
# question = bos_token + question
phi2 = phi2.to("cuda:0")

model_ans = phi2.calc_loss(question, collect_answer)
# %%
print(question)
print(model_ans.logits.shape)

model_ans_list = torch.argmax(model_ans.logits, dim=-1).tolist()[0]
print(phi2.tokenizer.decode(model_ans_list))
# %
print(phi2.tokenizer.special_tokens_map)
print(phi2.tokenizer.eos_token)
bos_token = phi2.tokenizer.bos_token
eos_token = phi2.tokenizer.eos_token

token_ids = phi2.tokenizer(
        question,
        padding="max_length",
        truncation=True,
        # max_length=20,
        return_tensors="pt",
    )
print(token_ids)
print(phi2.tokenizer.decode(token_ids["input_ids"][0]))
# %%

question = "What is the sum of 1 and 2?"
phi2.tokenizer(
    text=[[question], [question + "3"]],
    padding="longest",
    return_tensors="pt",
    is_split_into_words=True,
    add_special_tokens=True,
    return_attention_mask=True,
    return_length=True,
)
#%%
example = {'q_id': '7h191n',
 'title': 'What does the tax bill that was passed today mean? How will it affect Americans in each tax bracket?',
 'selftext': '',
 'category': 'Economics',
 'subreddit': 'explainlikeimfive',
 'answers.a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
  'answers.text': [''],
#  'answers.text': ["The tax bill is 500 pages long and there were a lot of changes still going on right to the end. It's not just an adjustment to the income tax brackets, it's a whole bunch of changes. As such there is no good answer to your question. The big take aways are: - Big reduction in corporate income tax rate will make large companies very happy. - Pass through rate change will make certain styles of business (law firms, hedge funds) extremely happy - Income tax changes are moderate, and are set to expire (though it's the kind of thing that might just always get re-applied without being made permanent) - People in high tax states (California, New York) lose out, and many of them will end up with their taxes raised.",
#   'None yet. It has to be reconciled with a vastly different house bill and then passed again.',
#   'Also: does this apply to 2017 taxes? Or does it start with 2018 taxes?',
#   'This article explains both the House and senate bills, including the proposed changes to your income taxes based on your income level. URL_0'],
 'answers.score': [21, 19, 5, 3],
 'answers.text_urls': [[],
  [],
  [],
  ['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']],
 'title_urls': ['url'],
 'selftext_urls': ['url']}

join_text = [" ".join(ans) for ans in example["answers.text"]]
print(join_text)
example = phi2.tokenizer(join_text)
# concatenated_examples = {k: sum(example[k], []) for k in example.keys()}
concatenated_examples = {k: example[k] for k in example.keys()}
print(concatenated_examples.keys())
print(concatenated_examples["input_ids"][0])
print(concatenated_examples["attention_mask"])

# %%
