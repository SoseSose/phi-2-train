# %%
from dataclasses import dataclass
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from settings_arc import CH_source
from pathlib import Path
import pytest


def plot_one(ax, input, show_num=False):
    """画像を一つ表示する関数

    Args:
        ax (_type_): _description_
        input (_type_): _description_
        show_num (bool, optional):動作検証用
    """
    color_list = [
        "#FFFFFF",
        "#000000",
        "#0074D9",
        "#FF4136",
        "#2ECC40",
        "#FFDC00",
        "#FFD700",
        "#F012BE",
        "#FF851B",
        "#7FDBFF",
        "#870C25",
        "#808080",
    ]
    cmap = colors.ListedColormap(color_list)
    norm = colors.Normalize(0, CH_source)
    ax.imshow(input, cmap=cmap, norm=norm)

    shape = input.shape

    ax.tick_params(length=0)
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + shape[0])])
    ax.set_xticks([x - 0.5 for x in range(1 + shape[1])])

    if show_num == True:
        for x in range(shape[1]):
            for y in range(shape[0]):
                plt.text(x, y, str(input[y, x]), size=6)

    ax.set_xticklabels([])
    ax.set_yticklabels([])


def plot_some(
    input: Union[np.ndarray, list],
    title: str,
    fold: int = 1,
    pad=None,
    save_file_name=None,
    show_num=False,
):
    """
    いくつかまとめて画像を表示する関数

    Args:
        input (np.ndarray):[N,H,W,C] or [N,H,W]のnp.ndarrayかlist, [N,H,W,C]の場合は最終次元でargmaxされる。
        title (str): 画像の上部に表示するテキスト
        fold (int): 画像を何行に分割するか. Defaults to 1.
        pad (bool, optional): None or [pad height, w_height]. Defaults to None. もし値がある時はpad_height, pad_widthの値までpaddingされる。
        save_file_name (_type_, optional):None or str. Defaults to None. 画像を保存する場合は保存場所と名前を指定する。
        show_num (bool): Defaults to False. 画像に数字を表示するかどうか。
    """
    if type(input) != np.ndarray:
        input = np.array(input)

    input_dim = len(input.shape)
    if not 3 <= input_dim <= 4:
        raise ValueError("input must be [N, H, W, C] or [N, H, W]")
    elif input_dim == 4:
        input = np.argmax(input, axis=-1)

    input_len = len(input)
    w = fold
    h = input_len // fold + 1
    fig = plt.figure(figsize=(3 * w, 3 * h))
    for i, val in enumerate(input):
        ax = fig.add_subplot(
            h,
            w,
            i + 1,
            title=title + ":" + str(i),
        )
        if pad != None:
            shape = val.shape
            val = np.pad(
                val,
                [[0, pad[0] - shape[0]], [0, pad[1] - shape[1]]],
                constant_values=0,
            )
        plot_one(ax, val, show_num)

    if save_file_name == None:
        plt.show()
    else:
        plt.savefig(save_file_name)
        plt.close()


def plot_task(
    train,
    test,
    candidate=None,
    model_answer=None,
    fold=1,
):
    train_inputs = train_outputs = []
    for train_inout in train:
        train_inputs.append(train_inout["input"])
        train_outputs.append(train_inout["output"])

    plot_some(train_inputs, "train input", fold=fold)
    plot_some(train_outputs, "train output", fold=fold)

    test_inputs = test_outputs = []
    for test_inout in test:
        test_inputs.append(test_inout["input"])
        test_outputs.append(test_inout["output"])

    plot_some(test_inputs, "test input", fold=fold)
    plot_some(test_outputs, "test output", fold=fold)

    if candidate != None:
        plot_some(candidate, "candidate", fold=fold)

    if model_answer != None:
        plot_some(model_answer, "model answer", fold=fold)



# %%


