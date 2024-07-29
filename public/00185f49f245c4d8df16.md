---
title: transformersのモデルのダウンロードについて
tags:
  - Python
  - transformers
private: false
updated_at: '2024-02-22T21:43:14+09:00'
id: 00185f49f245c4d8df16
organization_url_name: null
slide: false
ignorePublish: false
---
## 目的  

transformersのモデルをローカルにダウンロードする方法を示す。

## 動機

transformersで作業を続けているとき、アップデートが
入ると再ダウンロードが必要になる。Phi-2で約5GBの保存容量が必要になり
ダウンロードの度にそれなりの時間がかかり煩わしい。(光回線が引けない家🥺)

## ダウンロードするためのコード

AutoTokenizerやAutoModelForCausalLMのsave_pretrained関数[[1]](#save_pretrained関数について)のsave_directoryという引数に
保存先を指定するとモデルを指定場所に保存できる。
保存したあとはfrom_pretrainedのpretrained_model_name_or_path引数に保存した場所を指定すれば
モデルを読み込み使用できる。

```python

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_tokenizer_and_model(pre_trained_model_name, save_dir)
    if not Path(save_dir).exists():
        # download model
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(pre_trained_model_name)
        # save model
        self.tokenizer.save_pretrained(save_dir)
        self.model.save_pretrained(save_dir)

    else:
        # load model
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        model = AutoModelForCausalLM.from_pretrained(save_dir)

```  

## "the attention mask and the .."という警告

しかし、ダウンロードしたモデルで推論させようとすると次のような警告が表示される。

```txt
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
```

これはモデルにパディングトークンが渡されていないために発生する。
通常はパディングトークン=EOSトークンとして扱ってしまって問題ないので
次のコードのようにpad_token_idにtokenizer.eos_token_idを設定する。[[2]](#the-attention-mask-and-the-という警告に関する質問)

```python
model.generate(**encoded_input, pad_token_id=tokenizer.eos_token_id)
```

そうすることで上記の警告は発生しないように。

## まとめ

時々入る再ダウンロードに煩わされることがなく快適に!!

## 参考サイト

### save_pretrained関数について

https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained

### "the attention mask and the .."という警告に関する質問

https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id/71397707#71397707
