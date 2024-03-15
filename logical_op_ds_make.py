# %%
from operator import is_
import random
from re import I
import numpy as np
import numpy.typing as npt
from typing import Optional
from arc_preprocess import CH_source, MAX_SIZE, ArcImage
import pytest


def point_check(val: int):
    if val < 0:
        raise ValueError("Point val must be > 0")
    if val >= MAX_SIZE:
        raise ValueError("Point val must be < {}(MAX SIZE)".format(MAX_SIZE))


def from_to_check(from_val: int, to_val: int):
    if from_val >= to_val:
        raise ValueError("from_val must be less than to_val")


def paste(
    from_img: npt.NDArray[np.int32],
    to_img: npt.NDArray[np.int32],
    point_x: int,
    point_y: int,
) -> npt.NDArray[np.int32]:
    point_check(point_x)
    point_check(point_y)
    from_img_y, from_img_x = from_img.shape[:2]
    to_img_y, to_img_x = to_img.shape[:2]

    if from_img_x < point_x + to_img_x:
        raise ValueError("from_img.x > point_x + to_img.x")

    if from_img_y < point_y + to_img_y:
        raise ValueError("from_img.y > point_y + to_img.y")

    rslt_img = from_img.copy()
    rslt_img[point_y : point_y + to_img_y, point_x : point_x + to_img_x] = to_img

    return rslt_img


def test_point_val_annual():

    from_img = np.array([[1 for _ in range(MAX_SIZE)] for _ in range(MAX_SIZE)])
    to_img = np.array([[0 for _ in range(2)] for _ in range(2)])
    point_x = MAX_SIZE
    point_y = 0

    # check point_x lager than MAX_SIZE
    with pytest.raises(Exception) as e:
        rslt = paste(from_img, to_img, point_x, point_y)
    assert str(e.value) == f"Point val must be < {MAX_SIZE}(MAX SIZE)"

    # check point_y lager than MAX_SIZE
    point_x = 0
    point_y = MAX_SIZE
    with pytest.raises(Exception) as e:
        rslt = paste(from_img, to_img, point_x, point_y)
    assert str(e.value) == f"Point val must be < {MAX_SIZE}(MAX SIZE)"

    # check_point_x negative
    point_x = -1
    point_y = 1
    with pytest.raises(Exception) as e:
        rslt = paste(from_img, to_img, point_x, point_y)
    assert str(e.value) == f"Point val must be > 0"

    # check_point_y negative
    point_x = 1
    point_y = -1
    with pytest.raises(Exception) as e:
        rslt = paste(from_img, to_img, point_x, point_y)
    assert str(e.value) == f"Point val must be > 0"


def test_paint_spill_oper():
    # from_imgに指定されたポイントで貼り付けるとはみ出る場合をテストする

    from_img = np.array([[1 for _ in range(4)] for _ in range(4)])
    to_img = np.array([[0 for _ in range(2)] for _ in range(2)])

    # it should be ok
    point_x = 0
    point_y = 0
    rslt_img = paste(from_img, to_img, point_x, point_y)

    # it also should be ok
    point_x = 2
    point_y = 2
    rslt_img = paste(from_img, to_img, point_x, point_y)

    # it should be error on x
    point_x = 3
    point_y = 2
    with pytest.raises(Exception) as e:
        rslt_img = paste(from_img, to_img, point_x, point_y)
    assert str(e.value) == "from_img.x > point_x + to_img.x"

    # it should be error on y
    point_x = 2
    point_y = 3
    with pytest.raises(Exception) as e:
        rslt_img = paste(from_img, to_img, point_x, point_y)
    assert str(e.value) == "from_img.y > point_y + to_img.y"


def test_not_change_from_img_and_to_img():
    from_img = np.array([[1 for _ in range(4)] for _ in range(4)])
    to_img = np.array([[0 for _ in range(2)] for _ in range(2)])

    save_from_img = from_img.copy()
    save_to_img = to_img.copy()

    point_x = 0
    point_y = 0
    rslt_img = paste(from_img, to_img, point_x, point_y)

    assert (from_img == save_from_img).all()
    assert (to_img == save_to_img).all()


def draw_box(
    from_img: npt.NDArray[np.int32],
    x_from: int,
    x_to: int,
    y_from: int,
    y_to: int,
    fill_val: int,
) -> npt.NDArray[np.int32]:

    input_x, input_y = from_img.shape[:2]

    point_check(x_from)
    point_check(x_to)
    point_check(y_from)
    point_check(y_to)

    from_to_check(x_from, x_to)
    from_to_check(y_from, y_to)

    from_img[x_from:x_to, y_from:y_to] = fill_val
    return from_img


def test_draw_box():
    input = np.zeros((5, 5), dtype=int)
    rslt = draw_box(input, 1, 4, 1, 4, 1)
    assert (
        rslt
        == np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
    ).all()


class RandomLogicalOp:
    def __init__(
        self,
        a_0_b_0: Optional[bool] = None,
        a_0_b_1: Optional[bool] = None,
        a_1_b_0: Optional[bool] = None,
        a_1_b_1: Optional[bool] = None,
    ):

        if a_0_b_0 is not None:
            self.a_0_b_0 = a_0_b_0
        else:
            self.a_0_b_0 = random.getrandbits(1)

        if a_0_b_1 is not None:
            self.a_0_b_1 = a_0_b_1
        else:
            self.a_0_b_1 = random.getrandbits(1)

        if a_1_b_0 is not None:
            self.a_1_b_0 = a_1_b_0
        else:
            self.a_1_b_0 = random.getrandbits(1)

        if a_1_b_1 is not None:
            self.a_1_b_1 = a_1_b_1
        else:
            self.a_1_b_1 = random.getrandbits(1)

    def calc(self, a: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:

        if a.shape != b.shape:
            raise ValueError("a and b must have the same shape")

        # if ((a != True) & (a != 1)).all():
        if not np.issubdtype(a.dtype, np.bool_):
            raise ValueError("a should be bool val")

        if not np.issubdtype(b.dtype, np.bool_):
            raise ValueError("b should be bool val")

        rslt = np.where(
            (a == 0) & (b == 0),
            self.a_0_b_0,
            np.where(
                (a == 0) & (b == 1),
                self.a_0_b_1,
                np.where((a == 1) & (b == 0), self.a_1_b_0, self.a_1_b_1),
            ),
        )

        return rslt.astype(bool)


def test_RandomLogicalOp_and():

    and_logical_op = RandomLogicalOp(
        a_0_b_0=False,
        a_0_b_1=False,
        a_1_b_0=False,
        a_1_b_1=True,
    )

    a = np.array([[0, 1], [0, 1]], dtype=bool)
    b = np.array([[0, 1], [1, 0]], dtype=bool)
    rslt = and_logical_op.calc(a, b)
    assert (rslt == np.array([[0, 1], [0, 0]], dtype=bool)).all()


def test_RandomLogicalOp_normal_case():

    # check calc output is 0 or 1
    for _ in range(100):
        random_logical_op = RandomLogicalOp()

        a = np.array([[0 for _ in range(2)] for _ in range(2)], dtype=bool)
        b = np.array([[0 for _ in range(2)] for _ in range(2)], dtype=bool)
        rslt = random_logical_op.calc(a, b)
        assert ((rslt == 0) | (rslt == 1)).all()

        a = np.array([[0 for _ in range(2)] for _ in range(2)], dtype=bool)
        a = np.array([[0 for _ in range(2)] for _ in range(2)], dtype=bool)
        rslt = random_logical_op.calc(a, b)
        assert ((rslt == 0) | (rslt == 1)).all()

        a = np.array([[0 for _ in range(2)] for _ in range(2)], dtype=bool)
        a = np.array([[0 for _ in range(2)] for _ in range(2)], dtype=bool)
        rslt = random_logical_op.calc(a, b)
        assert ((rslt == 0) | (rslt == 1)).all()

        a = np.array([[1 for _ in range(2)] for _ in range(2)], dtype=bool)
        a = np.array([[1 for _ in range(2)] for _ in range(2)], dtype=bool)
        rslt = random_logical_op.calc(a, b)
        assert ((rslt == 0) | (rslt == 1)).all()


def test_RandomLogicalOp_ab_annual_val():

    random_logical_op = RandomLogicalOp()
    # case a is not 0 or 1
    a = np.array([[2 for _ in range(2)] for _ in range(2)], dtype=int)
    b = np.array([[0 for _ in range(2)] for _ in range(2)], dtype=bool)
    with pytest.raises(Exception) as e:
        random_logical_op.calc(a, b)
    assert str(e.value) == "a should be bool val"

    # case b is not 0 or 1
    a = np.array([[0 for _ in range(2)] for _ in range(2)], dtype=bool)
    b = np.array([[2 for _ in range(2)] for _ in range(2)], dtype=int)
    with pytest.raises(Exception) as e:
        random_logical_op.calc(a, b)
    assert str(e.value) == "b should be bool val"


MAX_PART_IMG_SIZE = int(MAX_SIZE / 2) - 1


def two_img_concat_with_line(
    img1: npt.NDArray[np.int32],
    img2: npt.NDArray[np.int32],
    line_color: int,
) -> npt.NDArray[np.int32]:
    # img1とimg2をline_colorで区切った画像を生成する。
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must have the same shape")

    if img1.shape[0] > MAX_PART_IMG_SIZE or img1.shape[1] > MAX_PART_IMG_SIZE:
        raise ValueError(f"img1 and img2 must be smaller than {MAX_PART_IMG_SIZE + 1}(MAX_PART_IMG_SIZE)")
    img1_y, img1_x = img1.shape[:2]

    rslt_img = np.full((img1_y * 2 + 1, img1_x), line_color, dtype=int)
    rslt_img = paste(rslt_img, img1, 0, 0)
    rslt_img = paste(rslt_img, img2, 0, img1_y + 1)

    return rslt_img


def test_two_img_concat_with_line():
    img1 = np.array([[1, 1], [1, 1]])
    img2 = np.array([[0, 0], [0, 0]])
    rslt = two_img_concat_with_line(img1, img2, 8)


def test_img1_and_img2_must_have_the_same_shape():
    img1 = np.array([[1, 1], [1, 1]])
    img2 = np.array([[0, 0, 0], [0, 0, 0]])
    with pytest.raises(Exception) as e:
        rslt = two_img_concat_with_line(img1, img2, 8)
    assert str(e.value) == "img1 and img2 must have the same shape"


def test_img1_and_img2_must_be_smaller_than_MAX_PART_IMG_SIZE():
    big_img = [[1 for _ in range(MAX_PART_IMG_SIZE)] for _ in range(MAX_PART_IMG_SIZE + 1)]

    img1 = np.array(big_img)
    img2 = np.array(big_img)
    with pytest.raises(Exception) as e:
        rslt = two_img_concat_with_line(img1, img2, 8)
    assert str(e.value) == f"img1 and img2 must be smaller than {MAX_PART_IMG_SIZE+1}(MAX_PART_IMG_SIZE)"

    big_img = [[1 for _ in range(MAX_PART_IMG_SIZE + 1)] for _ in range(MAX_PART_IMG_SIZE)]

    img1 = np.array(big_img)
    img2 = np.array(big_img)
    with pytest.raises(Exception) as e:
        rslt = two_img_concat_with_line(img1, img2, 8)
    assert str(e.value) == f"img1 and img2 must be smaller than {MAX_PART_IMG_SIZE+1}(MAX_PART_IMG_SIZE)"


def make_random_box(size_x: int, size_y: int, cand_val: list[int]):
    return np.random.choice(cand_val, size=(size_y, size_x))


def test_make_random_box():
    rslt = make_random_box(2, 2, [0, 1])
    assert rslt.shape == (2, 2)
    assert ((rslt == 0) | (rslt == 1)).all()


class Color:
    def __init__(self):
        self.color_cand = [i for i in range(CH_source)]

    def pick_random_unused(self, index: Optional[int] = None) -> int:
        if index is not None:
            if index < 0 or index >= len(self.color_cand):
                raise Exception(f"index must be 0 <= index < {len(self.color_cand)}")
            picked = self.color_cand.pop(index)
        else:
            picked = self.color_cand.pop(random.randint(0, len(self.color_cand) - 1))
        return picked


def test_color_pick():
    for _ in range(1000):
        color = Color()
        first_picked_color = color.pick_random_unused(index=None)
        second_picked_color = color.pick_random_unused(index=None)
        assert first_picked_color != second_picked_color


def test_all_color_pick():
    color = Color()
    for i in range(CH_source):
        picked_color = color.pick_random_unused(index=0)
        assert picked_color == i

    # it should be error
    color = Color()
    with pytest.raises(Exception) as e:
        picked_color = color.pick_random_unused(index=-1)
    assert str(e.value) == f"index must be 0 <= index < {CH_source}"

    # it should not be
    color = Color()
    picked_color = color.pick_random_unused(index=CH_source - 1)

    # it should be error
    color = Color()
    with pytest.raises(Exception) as e:
        picked_color = color.pick_random_unused(index=CH_source)
    assert str(e.value) == f"index must be 0 <= index < {CH_source}"


class ColorConverter:
    def __init__(self, zero_color: int, one_color: int):
        self.zero_color = zero_color
        self.one_color = one_color

    def to_binary(self, img: npt.NDArray[np.int32]):

        if ((img != self.zero_color) & (img != self.one_color)).all():
            raise ValueError(f"img val should be {self.zero_color} or {self.one_color}")

        rslt = np.where(img == self.zero_color, False, True)
        return rslt.astype(bool)

    def to_color(self, img: npt.NDArray[np.bool_]):
        if img.dtype != np.bool_:
            raise ValueError("img val should be bool val")

        return np.where(img == 0, self.zero_color, self.one_color)


class TestColorConverter:
    zero_color = 1
    one_color = 2
    color_converter = ColorConverter(zero_color=zero_color, one_color=one_color)

    def test_to_binary(self):
        zero_color = self.zero_color
        one_color = self.one_color
        img = np.array([[one_color, zero_color], [zero_color, one_color]])
        bin_img = self.color_converter.to_binary(img)
        assert bin_img.shape == (2, 2)
        assert (bin_img == np.array([[True, False], [False, True]])).all()

    def test_to_color(self):
        zero_color = self.zero_color
        one_color = self.one_color
        img = np.array([[True, False], [False, True]])
        color_img = self.color_converter.to_color(img)
        assert color_img.shape == (2, 2)
        assert (color_img == np.array([[one_color, zero_color], [zero_color, one_color]])).all()


def logical_out_img(
    img1: npt.NDArray[np.int32],
    img2: npt.NDArray[np.int32],
    color_converter: ColorConverter,
    logical_op: RandomLogicalOp,
) -> npt.NDArray[np.int32]:
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must have the same shape")

    img1_binary = color_converter.to_binary(img1)
    img2_binary = color_converter.to_binary(img2)

    rslt_img = logical_op.calc(img1_binary, img2_binary)
    out_img = color_converter.to_color(rslt_img)

    return out_img


def test_logical_out_img():
    zero_color = 3
    one_color = 8
    img1 = np.array([[zero_color, one_color], [zero_color, one_color]])
    img2 = np.array([[zero_color, one_color], [one_color, zero_color]])
    color_converter = ColorConverter(
        zero_color=zero_color,
        one_color=one_color,
    )
    logical_op = RandomLogicalOp(
        a_0_b_0=False,
        a_0_b_1=False,
        a_1_b_0=False,
        a_1_b_1=True,
    )
    rslt = logical_out_img(img1, img2, color_converter, logical_op)
    assert rslt.shape == (2, 2)
    assert (rslt == np.array([[zero_color, one_color], [zero_color, zero_color]])).all()

    # img1 and img2 must have the same shape
    zero_color = 3
    one_color = 8
    img1 = np.array([[zero_color, one_color], [zero_color, one_color]])
    img2 = np.array([[zero_color], [one_color]])
    with pytest.raises(Exception) as e:
        rslt = logical_out_img(img1, img2, color_converter, logical_op)
    assert str(e.value) == "img1 and img2 must have the same shape"


if __name__ == "__main__":
    test_logical_out_img()


def logical_inout_img(
    logical_op: RandomLogicalOp,
    zero_color: int,
    one_color: int,
    line_color: int,
    is_vertical: bool,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:

    size_x = random.randint(1, MAX_PART_IMG_SIZE)
    size_y = random.randint(1, MAX_PART_IMG_SIZE)

    img1 = make_random_box(size_y=size_y, size_x=size_x, cand_val=[zero_color, one_color])
    img2 = make_random_box(size_y=size_y, size_x=size_x, cand_val=[zero_color, one_color])

    in_img = two_img_concat_with_line(
        img1=img1,
        img2=img2,
        line_color=line_color,
    )

    color_converter = ColorConverter(zero_color, one_color)
    out_img = logical_out_img(
        img1=img1,
        img2=img2,
        color_converter=color_converter,
        logical_op=logical_op,
    )

    if is_vertical:
        in_img = in_img.T
        out_img = out_img.T

    return in_img, out_img


def test_logical_inout_img():
    zero_color = 1
    one_color = 3
    line_color = 4
    in_img, out_img = logical_inout_img(
        logical_op=RandomLogicalOp(),
        zero_color=zero_color,
        one_color=one_color,
        line_color=line_color,
        is_vertical=False,
    )
    in_img_y = in_img.shape[0]
    part_img_size = int(in_img_y / 2) - 1

    img1 = in_img[: part_img_size + 1]
    line = in_img[part_img_size + 1]
    img2 = in_img[part_img_size + 2 :]

    # img1, img2 shold be zero_color or one_color
    assert ((img1 == zero_color) | (img1 == one_color)).all()
    assert ((img2 == zero_color) | (img2 == one_color)).all()
    # line should be line_color
    assert (line == line_color).all()
    # out_img should be zero_color or one_color
    assert ((out_img == zero_color) | (out_img == one_color)).all()


def logical_op_img_decomp(
    img: npt.NDArray[np.int32], is_vertical: bool
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:

    if is_vertical:
        img = img.T

    img_y, img_x = img.shape[:2]
    part_img_size = int(img_y / 2) - 1

    img1 = img[: part_img_size + 1]
    line = img[part_img_size + 1]
    img2 = img[part_img_size + 2 :]

    return img1, line, img2


def test_logical_inout_img_is_vertical():
    zero_color = 1
    one_color = 3
    line_color = 4
    in_img, out_img = logical_inout_img(
        logical_op=RandomLogicalOp(),
        zero_color=zero_color,
        one_color=one_color,
        line_color=line_color,
        is_vertical=True,
    )
    img1, line, img2 = logical_op_img_decomp(in_img, is_vertical=True)

    # img1, img2 shold be zero_color or one_color
    assert ((img1 == zero_color) | (img1 == one_color)).all()
    assert ((img2 == zero_color) | (img2 == one_color)).all()
    # line should be line_color
    assert (line == line_color).all()
    # out_img should be zero_color or one_color
    assert ((out_img == zero_color) | (out_img == one_color)).all()


def logical_inout_task(task_len: int) -> tuple[list[npt.NDArray[np.int32]], list[npt.NDArray[np.int32]]]:

    logical_op = RandomLogicalOp()
    color = Color()
    zero_color = color.pick_random_unused()
    one_color = color.pick_random_unused()
    line_color = color.pick_random_unused()
    is_vertical = random.randint(0, 1) == 0
    in_imgs = []
    out_imgs = []

    for _ in range(task_len):
        in_img, out_img = logical_inout_img(logical_op, zero_color, one_color, line_color, is_vertical)
        in_imgs.append(in_img)
        out_imgs.append(out_img)

    return in_imgs, out_imgs


def test_logical_inout_task():
    in_imgs, out_imgs = logical_inout_task(10)
    in_img = in_imgs[0]
    color_num_in_row = np.unique(in_imgs[0][0]).shape[0]
    is_vertical = color_num_in_row == 3

    
    for i, in_img in enumerate(in_imgs):
        img1, line, img2 = logical_op_img_decomp(in_img, is_vertical)
        if i == 0:
            stacked_imgs = img1.flatten()
            stacked_lines = line.flatten()
        else:
            stacked_imgs = np.concatenate((stacked_imgs, img1.flatten()))
            stacked_lines = np.concatenate((stacked_lines, line.flatten()))
        stacked_imgs = np.concatenate((stacked_imgs, img2.flatten()))
        assert img1.shape == img2.shape

    assert np.unique(stacked_imgs).shape[0] == 2
    assert np.unique(stacked_lines).shape[0] == 1

    for i, out_img in enumerate(out_imgs):
        if i == 0:
            stacked_out_imgs = out_img.flatten()
        else:
            stacked_out_imgs = np.concatenate((stacked_out_imgs, out_img.flatten()))
    
    assert np.unique(stacked_out_imgs).shape[0] == 2
        

if __name__ == "__main__":
    test_logical_inout_task()
#%%


def np_img_to_arc_img(img: npt.NDArray[np.int32]) -> ArcImage:
    return ArcImage(img.tolist())


from pathlib import Path
import uuid
import json


#!! テストしてない
def save_logical_task(dir: str):
    path_path = Path(dir)  # type: ignore
    in_imgs, out_imgs = logical_inout_task(10)
    task = {
        "in_imgs": in_imgs,
        "out_imgs": out_imgs,
        # "logical_op": logical_op.__dict__,
    }
    file_name = str(uuid.uuid4()) + ".json"
    with open(path_path / file_name, "w") as file:
        json.dump(task, file)
    return file_name


def load_logical_task(dir: str, f_name: str) -> dict:
    path_path = Path(dir)  # type: ignore
    with open(path_path / f_name, "r") as file:
        task = json.load(file)
    return task


# def test_


def save_logical_tasks(dir: str, task_num: int):
    pass


def load_logical_tasks(dir_path: str):

    path = Path("result/logical_task")
    if not path.exists():
        path.mkdir()
    files = path.glob("*.json")
    tasks = []
    for file in files:
        with open(file, "r") as file:
            task = json.load(file)
        tasks.append(task)
    return tasks


if __name__ == "__main__":
    from arc_visualize import plot_task

    and_logical_op = RandomLogicalOp(
        a_0_b_0=False,
        a_0_b_1=False,
        a_1_b_0=False,
        a_1_b_1=True,
    )

    in_imgs, out_imgs = logical_inout_task(
        10,
        # logical_op=and_logical_op,
    )
    print(out_imgs[1])

    plot_task(
        train_inputs=in_imgs,
        train_outputs=out_imgs,
        test_inout=[in_imgs[0], out_imgs[0]],
        save_path="result/test.png",
    )

