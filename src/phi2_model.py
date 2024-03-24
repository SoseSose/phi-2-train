# %%
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from abc import abstractclassmethod, ABC


class BaseModel(ABC):
    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def build(self):
        pass

    @abstractclassmethod
    def get_token_num_and_answer(self, question:str)->tuple[str, int]:
        pass


class Phi2(BaseModel):
    MAX_TOKENS = 2048

    def __init__(self, save_dir, device="auto"):
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

    def get_token_num_and_answer(self, question:str)->tuple[str, int]:
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
