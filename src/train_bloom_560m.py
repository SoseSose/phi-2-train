# %%
import torch
from architectures.bloom_560m import get_bloom560m_tokenizer, Bloom560m 
from data_processing.easy_ds_EN_to_SP import  get_masked_ds,predict_training_set, try_print_iterative_gen, EasyEnToSpDM
from lightning import Trainer

torch.set_float32_matmul_precision('high')
#You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precisionというエラーを避ける

#%%
tokenizer = get_bloom560m_tokenizer("D:/models")

#%%
easy_en_to_sp_dm = EasyEnToSpDM(tokenizer, 1)
model = Bloom560m("D:/models", 2e-4)

#%%
predict_training_set(model.model, tokenizer)
try_print_iterative_gen(model.model, tokenizer)
#%%
for data in easy_en_to_sp_dm.ds:
    print(data)

#%%
trainer = Trainer(
    # fast_dev_run=True,
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
)

trainer.fit(
    model=model,
    datamodule=easy_en_to_sp_dm,
)

# %%
def test_get_masked_ds():
    # sanity check that our format is correct
    # we'd expect -100 for the human text and the actual token(s) for the assistant text
    masked_dataset = get_masked_ds(tokenizer)
    label_ex = masked_dataset[0]["labels"]
    print(f"{label_ex=}")
    # let's see just the non-masked text
    non_masked_text = tokenizer.decode(label_ex[label_ex != -100], skip_special_tokens=False)
    assert non_masked_text == " perro</s>"
    print(f"non masked text: {non_masked_text}")
    # let's see just the masked text
    # -100 is not a real token, convert to something the tokenizer understands
    label_ex[label_ex == -100] = 0
    full_lable = tokenizer.decode(label_ex, skip_special_tokens=False)
    print(f"full 'label': {full_lable}")
    assert full_lable == "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> perro</s>"