# %%
from dataclasses import dataclass
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from typing import Union


@dataclass
class Plotter:
    def _plot_one(self, ax, input, show_num=False):
        """_summary_

        Args:
            ax (_type_): _description_
            input (_type_): _description_
            show_num (bool, optional):動作検証用

        Returns:
            _type_: _description_
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
        cmap = colors.ListedColormap(color_list)  # type: ignore
        norm = colors.Normalize(0, 10)  # magic number
        ax.imshow(input, cmap=cmap, norm=norm)

        shape = input.shape

        ax.tick_params(length=0)
        ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
        ax.set_yticks([x - 0.5 for x in range(1 + shape[0])])
        ax.set_xticks([x - 0.5 for x in range(1 + shape[1])])

        if show_num == True:
            for x in range(shape[1]):
                for y in range(shape[0]):
                    plt.text(x, y, str(input[y, x]), size=5)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        return None

    def plot_some(
        self,
        input: Union[np.ndarray, list],
        title: str,
        fold: int = 1,
        pad=None,
        save_path=None,
        show_num=False,
    ):
        """_summary_

        Args:
            input (np.ndarray):[N,H,W,C] or [N,H,W]のnp.ndarrayかlist, [N,H,W,C]の場合は最終次元でargmaxされる。
            title (str): 画像の上部に表示するテキスト
            pad (bool, optional): None or [pad height, w_height]. Defaults to None. もし値がある時はpad_height, pad_widthの値までpaddingされる。
            save_path (_type_, optional):None or str. Defaults to None. 画像を保存する場合はパスを指定する。
        """
        if type(input) == list:
            input = np.array(input)
        if not 3 <= len(input.shape) <= 4:
            raise ValueError("input must be [N, H, W, C] or [N, H, W]]")
        elif len(input.shape) == 4:
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
                title=title,
            )
            if pad != None:
                shape = val.shape
                val = np.pad(
                    val,
                    [[0, pad[0] - shape[0]], [0, pad[1] - shape[1]]],
                    constant_values=0,
                )
            self._plot_one(ax, val, show_num)

        if save_path == None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

    def plot_some_list(
        self, input: list, title: str, fold: int = 1, save_path=None, show_num=False
    ):
        input = list(map(lambda input: input.astype(np.int32), input))

        input_len = len(input)

        w = fold
        h = input_len // fold + 1
        fig = plt.figure(figsize=(3 * w, 3 * h))
        for i, val in enumerate(input):
            ax = fig.add_subplot(
                h,
                w,
                i + 1,
                title=title,
            )
            if type(val) != np.ndarray:
                val = np.array(val)
            self._plot_one(ax, val, show_num)

        if save_path == None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()


# %%
if __name__ == "__main__":
    CH_source = 10
    SIZE = 30
    height = 6
    plotter_instance = Plotter()
    original = np.tile(np.arange(CH_source + 2), (8, height, 1))
    plotter_instance.plot_some(
        original, "original" + str(original.shape), fold=5, show_num=True
    )
# %%
if __name__ == "__main__":
    CH_source = 10
    SIZE = 30
    height = 6
    plotter_instance = Plotter()

    original = np.tile(np.arange(CH_source + 2), (2, height, 1))
    # test normal plot
    plotter_instance.plot_some(
        original, "original" + str(original.shape), show_num=True
    )
    plotter_instance.plot_some(
        original + 4, "original" + str(original.shape), show_num=True
    )
    # test padded plot
    plotter_instance.plot_some(original, "padded image", pad=(SIZE, SIZE))

    # test one hot plot
    def one_hot(x: np.ndarray, depth):
        return np.identity(depth)[x]

    one_hot_img = one_hot(original, CH_source + 2)
    plotter_instance.plot_some(
        one_hot_img, "image" + str(one_hot_img.shape), pad=(SIZE, SIZE)
    )

    # test save path
    plotter_instance.plot_some(
        original,
        "test save",
        pad=(SIZE, SIZE),
        save_path="C:\\Users\\taeya\\Documents\\ARK\\data\\arc_log_img\\test.png",
    )

# %%
