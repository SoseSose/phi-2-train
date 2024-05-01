# %%
from copy import copy
from time import time
from typing import Dict, List, Optional

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from pathlib import Path


INSTRUCTION_TEMPLATE_BASE = "\n\n### Human:"
RESPONSE_TEMPLATE_BASE = "\n\n### Assistant:"

#%%

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")



# %%
def get_tokenizer(model_name: str, save_dir: str, **kwargs):
    save_path = Path(save_dir) /"tokenizer"/ model_name
    if Path(save_path).exists():
        tokenizer = AutoTokenizer.from_pretrained(
            save_path,
            **kwargs,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **kwargs,
        )
        tokenizer.save_pretrained(save_path)
    return tokenizer


def get_model(model_name: str, save_dir: str, **kwargs):
    save_path = Path(save_dir) / "model"/ model_name
    if Path(save_path).exists():
        model = AutoModelForCausalLM.from_pretrained(
            save_path,
            **kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        )
        model.save_pretrained(save_path)
    return model


# %%
def add_special_tokens(
    example: Dict,
    tokenizer: PreTrainedTokenizerBase,
) -> Dict:
    # add eos_token before human text and bos_token before assistant text
    example["text"] = (
        example["text"]
        .replace(
            INSTRUCTION_TEMPLATE_BASE, tokenizer.eos_token + INSTRUCTION_TEMPLATE_BASE
        )
        .replace(RESPONSE_TEMPLATE_BASE, RESPONSE_TEMPLATE_BASE + tokenizer.bos_token)
    )
    if not example["text"].endswith(tokenizer.eos_token):
        example["text"] += tokenizer.eos_token
    # Remove leading EOS tokens
    while example["text"].startswith(tokenizer.eos_token):
        example["text"] = example["text"][len(tokenizer.eos_token) :]

    return example


# Preprocessing
model_name = "bigscience/bloom-560m"
tokenizer = get_tokenizer(
    model_name, "D:/models", trust_remote_code=True, padding_side="right"
)
#%%
str1 = '\n\n### Human: How do you say "dog" in Spanish?\n\n### Assistant: perro'
str2 = '\n\n### Human: How do you say "water" in Spanish?\n\n### Assistant: agua'
str3 = '\n\n### Human: How do you say "mother" in Spanish?\n\n### Assistant: madre'
str4 = '\n\n### Human: How do you say "hello" in Spanish?\n\n### Assistant: hola'
str5 = '\n\n### Human: How do you say "tree" in Spanish?\n\n### Assistant: árbol'
train_data = {
    "text": [str1, str2, str3, str4, str5],
}
dataset_text = Dataset.from_dict(train_data)
dataset_text = dataset_text.map(lambda x: add_special_tokens(x, tokenizer))
for data in dataset_text:
    print(data)
# %%
# tokenize the text
dataset = dataset_text.map(
    lambda example: tokenizer(example["text"], padding=True), batched=True, remove_columns=["text"],
    #! padding=Trueを追加している.

)
# copy the input_ids to labels
dataset = dataset.map(lambda x: {"labels": x["input_ids"]}, batched=True)
print(f"{dataset=}")
for data in dataset:
    print(f"{data['input_ids']=}")
    print(len(data['input_ids']))
    # print(f"{data['labels']=}")
    # print(len(data['labels']))
    # print(f"{data['attention_mask']=}")
    # print(len(data['attention_mask']))

# %%
print_gpu_utilization()
#%%

def load_model(
    model_name: str, peft_kwargs: Optional[Dict] = None
) -> PeftModelForCausalLM:
    model = get_model(model_name, "D:/models")
    if peft_kwargs is None:
        peft_kwargs = {}
    peft_config = LoraConfig(task_type="CAUSAL_LM", **peft_kwargs)
    # alterantively, you can use the following to load the model
    # model = PeftModelForCausalLM.from_pretrained(model_name)
    model = get_peft_model(model, peft_config)
    return model
model = load_model(model_name)

print_gpu_utilization()

#%%
def sample_generate(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    inputs: BatchEncoding,
    **kwargs,
) -> str:
    """Runs tokenized text through the model and returns the generated text."""
    outputs = model.generate(**inputs, **kwargs)
    gen_text = tokenizer.batch_decode(
        # strip the text of the prompt
        outputs[:, inputs["input_ids"].shape[1] :]
    )
    return gen_text[0]


holdout_str = (
    '\n\n### Human: How do you say "good" in Spanish?\n\n### Assistant:<s>'  # bueno
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print_gpu_utilization()
#%%
holdout_input = tokenizer(holdout_str, return_tensors="pt").to(device)
original_output = sample_generate(model, tokenizer, holdout_input, max_new_tokens=5)
print(original_output)
print_gpu_utilization()

#%%
for data in dataset:

    data["input_ids"] = torch.tensor(data["input_ids"]).unsqueeze(0).to(device)
    data["attention_mask"] = torch.tensor(data["attention_mask"]).unsqueeze(0).to(device)
    data["labels"] = torch.tensor(data["labels"]).unsqueeze(0).to(device)
    out = model(**data)
print_gpu_utilization()

# %%
ENGLISH_WORDS = ["dog", "water", "mother", "hello", "tree"]
SPANISH_WORDS = ["perro", "agua", "madre", "hola", "árbol"]

def predict_training_set(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    english_words: List[str] = ENGLISH_WORDS,
    spanish_words: List[str] = SPANISH_WORDS,
):
    """Runs predictions on the entire training set."""
    for eng, span in zip(english_words, spanish_words):
        inputs2 = tokenizer(
            f'\n\n### Human: How do you say "{eng}" in Spanish?\n\n### Assistant:<s>',
            return_tensors="pt",
        ).to(device)
        print(
            "real answer:",
            span,
            "\tpredicted answer:",
            sample_generate(model, tokenizer, inputs2, max_new_tokens=5),
        )


predict_training_set(model, tokenizer)
# %%
def print_iterative_generate(model, tokenizer, inputs):
    """Approximates the training forward pass by iterating through a sequence
    and predicting one token at a time.
    """
    tok_outputs = []
    for tok_id in range(1, len(inputs["input_ids"][0]) + 1):
        iterative_inputs = inputs.copy()
        iterative_inputs["input_ids"] = inputs["input_ids"][:, :tok_id]
        iterative_inputs["attention_mask"] = inputs["attention_mask"][:, :tok_id]
        tok_outputs.append(
            sample_generate(model, tokenizer, iterative_inputs, max_new_tokens=1)
        )
    print("".join(tok_outputs))

print_iterative_generate(model, tokenizer, holdout_input)
# %%

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=500,
    per_device_train_batch_size=1,
    seed=1,
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_arguments,
)
training1 = trainer.train()
#%%
predict_training_set(model, tokenizer)
# %%
trainer.train()
#%%
predict_training_set(model, tokenizer)
# %%
sample_generate(model, tokenizer, holdout_input, max_new_tokens=5)
#%%
print_iterative_generate(model, tokenizer, holdout_input)
# %%
def create_special_mask(example: Dict) -> Dict:
    """Mask human text and keep assistant text as it is.

    Args:
        example (Dict): Result of tokenizing some text

    Returns:
        Dict: The dict with the label masked
    """
    # setting a token to -100 is how we "mask" a token
    # and tell the model to ignore it when calculating the loss
    mask_token_id = -100
    # assume we always start with a human text
    human_text = True
    for idx, tok_id in enumerate(example["labels"]):
        if human_text:
            # mask all human text up until and including the bos token
            example["labels"][idx] = mask_token_id
            if tok_id == tokenizer.bos_token_id:
                human_text = False
        elif not human_text and tok_id == tokenizer.eos_token_id:
            human_text = True
        elif not human_text:
            # leave example['labels'] text as it is when assistant text
            continue
    return example
#%%
# sanity check that our format is correct
# we'd expect -100 for the human text and the actual token(s) for the assistant text
# mask human characters but keep the assistant text as it is
dataset_masked = dataset.map(create_special_mask)
# convert dataset from lists to torch tensors
dataset_masked.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
print(f"{dataset_masked[0]["labels"]=}")
label_ex = dataset_masked[0]["labels"]
# let's see just the non-masked text
print(f"non masked text: {tokenizer.decode(label_ex[label_ex != -100], skip_special_tokens=False)}")
# let's see just the masked text
# -100 is not a real token, convert to something the tokenizer understands
label_ex[label_ex == -100] = 0
print(f"full 'label': {tokenizer.decode(label_ex, skip_special_tokens=False)}")

#%%
# Reset the model
model = load_model(model_name)
training2 = trainer.train()
#%%
print(f"{training2.metrics['train_runtime']=}")
print(f"{training1.metrics['train_runtime'] =}")
print(
    f"{100*round((training1.metrics['train_runtime']  - training2.metrics['train_runtime']) / training1.metrics['train_runtime'] , 2)}%"
)
#%%
predict_training_set(model, tokenizer)
sample_generate(model, tokenizer, holdout_input, max_new_tokens=5)
print_iterative_generate(model, tokenizer, holdout_input)