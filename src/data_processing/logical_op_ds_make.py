# %%
import dataclasses
import pickle
import random
import uuid
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from tqdm import tqdm

from data_processing.arc_preprocess import ArcImage, ArcInout, ArcTask

from data_processing.const import ArcConst, ArcColor

MIN_COLOR_NUM = ArcConst.MIN_COLOR_NUM
MAX_COLOR_NUM = ArcConst.MAX_COLOR_NUM
MAX_IMG_SIZE = ArcConst.MAX_IMG_SIZE
MIN_IMG_SIZE = ArcConst.MIN_IMG_SIZE
MAX_PART_IMG_SIZE = int(MAX_IMG_SIZE / 2) - 1

LOGICAL_OP_MIN_SIZE = 5


def point_check(val: int):
    if val > MAX_IMG_SIZE - 1:
        raise ValueError(
            f"Point val must be >= {MAX_IMG_SIZE-1} and <{MIN_IMG_SIZE-1}, but {val} is given"
        )
    elif val < MIN_IMG_SIZE - 1:
        raise ValueError(
            f"Point val must be >= {MAX_IMG_SIZE-1} and <{MIN_IMG_SIZE-1}, but {val} is given"
        )


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
    from_img = np.array([[1 for _ in range(MAX_IMG_SIZE)] for _ in range(MAX_IMG_SIZE)])
    to_img = np.array([[0 for _ in range(2)] for _ in range(2)])
    point_x = MAX_IMG_SIZE
    point_y = 0

    # check point_x lager than MAX_SIZE
    with pytest.raises(Exception) as e:
        _ = paste(from_img, to_img, point_x, point_y)
    assert (
        str(e.value)
        == f"Point val must be >= {MAX_IMG_SIZE-1} and <{MIN_IMG_SIZE-1}, but {point_x} is given"
    )

    # check point_y lager than MAX_SIZE
    point_x = 0
    point_y = MAX_IMG_SIZE
    with pytest.raises(Exception) as e:
        _ = paste(from_img, to_img, point_x, point_y)
    assert (
        str(e.value)
        == f"Point val must be >= {MAX_IMG_SIZE-1} and <{MIN_IMG_SIZE-1}, but {point_y} is given"
    )

    # check_point_x negative
    point_x = -1
    point_y = 1
    with pytest.raises(Exception) as e:
        _ = paste(from_img, to_img, point_x, point_y)
    assert (
        str(e.value)
        == f"Point val must be >= {MAX_IMG_SIZE-1} and <{MIN_IMG_SIZE-1}, but {point_x} is given"
    )

    # check_point_y negative
    point_x = 1
    point_y = -1
    with pytest.raises(Exception) as e:
        _ = paste(from_img, to_img, point_x, point_y)
    assert (
        str(e.value)
        == f"Point val must be >= {MAX_IMG_SIZE-1} and <{MIN_IMG_SIZE-1}, but {point_y} is given"
    )


def test_paint_spill_oper():
    # from_imgに指定されたポイントで貼り付けるとはみ出る場合をテストする

    from_img = np.array([[1 for _ in range(4)] for _ in range(4)])
    to_img = np.array([[0 for _ in range(2)] for _ in range(2)])

    # it should be ok
    point_x = 0
    point_y = 0
    _ = paste(from_img, to_img, point_x, point_y)

    # it also should be ok
    point_x = 2
    point_y = 2
    _ = paste(from_img, to_img, point_x, point_y)

    # it should be error on x
    point_x = 3
    point_y = 2
    with pytest.raises(Exception) as e:
        _ = paste(from_img, to_img, point_x, point_y)
    assert str(e.value) == "from_img.x > point_x + to_img.x"

    # it should be error on y
    point_x = 2
    point_y = 3
    with pytest.raises(Exception) as e:
        _ = paste(from_img, to_img, point_x, point_y)
    assert str(e.value) == "from_img.y > point_y + to_img.y"


def test_not_change_from_img_and_to_img():
    from_img = np.array([[1 for _ in range(4)] for _ in range(4)])
    to_img = np.array([[0 for _ in range(2)] for _ in range(2)])

    save_from_img = from_img.copy()
    save_to_img = to_img.copy()

    point_x = 0
    point_y = 0
    _ = paste(from_img, to_img, point_x, point_y)

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


class LogicalOp:
    def calc(
        self, a: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.bool_]:
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


class AllOneLogicalOp(LogicalOp):
    def __init__(self):
        self.a_0_b_0 = 1
        self.a_0_b_1 = 1
        self.a_1_b_0 = 1
        self.a_1_b_1 = 1


class LogicalAndOp(LogicalOp):
    def __init__(self):
        self.a_0_b_0 = 0
        self.a_0_b_1 = 0
        self.a_1_b_0 = 0
        self.a_1_b_1 = 1


class RandomLogicalOp(LogicalOp):
    def __init__(
        self,
    ):
        self.a_0_b_0 = random.getrandbits(1)
        self.a_0_b_1 = random.getrandbits(1)
        self.a_1_b_0 = random.getrandbits(1)
        self.a_1_b_1 = random.getrandbits(1)


def test_RandomLogicalOp_and():
    and_logical_op = LogicalAndOp()
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


class LogicalOpSttgs:
    def __init__(
        self,
        logical_op: LogicalOp,
        zero_color: int,
        one_color: int,
        line_color: int,
        is_vertical: bool,
        size_x: int,
        size_y: int,
    ):
        self.logical_op = logical_op
        self.zero_color = zero_color
        self.one_color = one_color
        self.line_color = line_color
        self.is_vertical = is_vertical
        self.size_x = size_x
        self.size_y = size_y


def two_img_concat_with_line(
    img1: npt.NDArray[np.int32],
    img2: npt.NDArray[np.int32],
    line_color: int,
) -> npt.NDArray[np.int32]:
    # img1とimg2をline_colorで区切った画像を生成する。
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must have the same shape")

    img1_y, img1_x = img1.shape[:2]
    if img1_y > MAX_PART_IMG_SIZE and img1_x > MAX_PART_IMG_SIZE:
        raise ValueError(f"img_y or img_x > {MAX_PART_IMG_SIZE}")

    rslt_img = np.full((img1_y * 2 + 1, img1_x), line_color, dtype=int)
    rslt_img = paste(rslt_img, img1, 0, 0)
    rslt_img = paste(rslt_img, img2, 0, img1_y + 1)

    return rslt_img


def test_two_img_concat_with_line():
    img1 = np.array([[1, 1], [1, 1]])
    img2 = np.array([[0, 0], [0, 0]])
    rslt = two_img_concat_with_line(img1, img2, 8)
    predicted_out = np.array([[1, 1], [1, 1], [8, 8], [0, 0], [0, 0]], dtype=np.int32)
    assert (rslt == predicted_out).all()


def test_img1_and_img2_must_have_the_same_shape():
    img1 = np.array([[1, 1], [1, 1]])
    img2 = np.array([[0, 0, 0], [0, 0, 0]])
    with pytest.raises(Exception) as e:
        _ = two_img_concat_with_line(img1, img2, 8)
    assert str(e.value) == "img1 and img2 must have the same shape"


def test_img1_and_img2_must_be_smaller_than_MAX_PART_IMG_SIZE():
    big_img = [
        [1 for _ in range(MAX_PART_IMG_SIZE + 1)] for _ in range(MAX_PART_IMG_SIZE + 1)
    ]

    img1 = np.array(big_img)
    img2 = np.array(big_img)
    with pytest.raises(Exception) as e:
        _ = two_img_concat_with_line(img1, img2, 8)
    assert str(e.value) == f"img_y or img_x > {MAX_PART_IMG_SIZE}"


def make_random_box(size_x: int, size_y: int, cand_val: list[int]):
    return np.random.choice(cand_val, size=(size_y, size_x))


def test_make_random_box():
    rslt = make_random_box(2, 2, [0, 1])
    assert rslt.shape == (2, 2)
    assert ((rslt == 0) | (rslt == 1)).all()


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
        assert (
            color_img == np.array([[one_color, zero_color], [zero_color, one_color]])
        ).all()


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
    logical_op = LogicalAndOp()
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


def logical_inout_img(
    logical_op: LogicalOp,
    zero_color: int,
    one_color: int,
    line_color: int,
    is_vertical: bool,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    size_y = random.randint(LOGICAL_OP_MIN_SIZE, MAX_PART_IMG_SIZE)
    size_x = random.randint(LOGICAL_OP_MIN_SIZE, MAX_IMG_SIZE)

    img1 = make_random_box(
        size_y=size_y, size_x=size_x, cand_val=[zero_color, one_color]
    )
    img2 = make_random_box(
        size_y=size_y, size_x=size_x, cand_val=[zero_color, one_color]
    )

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


def logical_op_img_decomp(
    img: npt.NDArray[np.int32], is_vertical: bool
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    if is_vertical:
        img = img.T

    img_y = img.shape[0]
    part_img_size = int(img_y / 2) - 1

    img1 = img[: part_img_size + 1]
    line = img[part_img_size + 1]
    img2 = img[part_img_size + 2 :]

    if is_vertical:
        img1 = img1.T
        line = line.T
        img2 = img2.T
    return img1, line, img2


def color_use_uncorrct(
    in_img: npt.NDArray[np.int32],
    out_img: npt.NDArray[np.int32],
    zero_color: int,
    one_color: int,
    line_color: int,
    is_vertical: bool,
):
    def assert_only_zero_or_one_color(img: npt.NDArray[np.int32]):
        assert ((img == zero_color) | (img == one_color)).all()

    img1, line, img2 = logical_op_img_decomp(in_img, is_vertical=is_vertical)

    # img1, img2 shold be zero_color or one_color
    assert_only_zero_or_one_color(img1)
    assert_only_zero_or_one_color(img2)
    # line should be line_color
    assert (line == line_color).all()
    # out_img should be zero_color or one_color
    assert_only_zero_or_one_color(out_img)


@pytest.mark.parametrize("is_vertical", [True, False])
def test_logical_inout_img(is_vertical):
    zero_color = 1
    one_color = 3
    line_color = 4
    is_vertical = False
    for _ in range(1000):
        in_img, out_img = logical_inout_img(
            logical_op=RandomLogicalOp(),
            zero_color=zero_color,
            one_color=one_color,
            line_color=line_color,
            is_vertical=is_vertical,
        )

        color_use_uncorrct(
            in_img, out_img, zero_color, one_color, line_color, is_vertical
        )


@dataclasses.dataclass
class RandomValues:
    logical_op = RandomLogicalOp()
    color = ArcColor()
    zero_color = color.pick_random_unused()
    one_color = color.pick_random_unused()
    line_color = color.pick_random_unused()
    is_vertical = random.randint(0, 1) == 0


def logical_inout_task(
    task_len: int, random_vals: RandomValues
) -> tuple[list[npt.NDArray[np.int32]], list[npt.NDArray[np.int32]]]:
    logical_op = random_vals.logical_op
    zero_color = random_vals.zero_color
    one_color = random_vals.one_color
    line_color = random_vals.line_color
    is_vertical = random_vals.is_vertical

    in_imgs = []
    out_imgs = []

    for _ in range(task_len):
        in_img, out_img = logical_inout_img(
            logical_op, zero_color, one_color, line_color, is_vertical
        )
        in_imgs.append(in_img)
        out_imgs.append(out_img)

    return in_imgs, out_imgs


def test_logical_inout_task():
    random_vals = RandomValues()
    in_imgs, out_imgs = logical_inout_task(10, random_vals=random_vals)

    for in_img, out_img in zip(in_imgs, out_imgs):
        color_use_uncorrct(
            in_img,
            out_img,
            random_vals.zero_color,
            random_vals.one_color,
            random_vals.line_color,
            random_vals.is_vertical,
        )


def np_img_to_arc_img(img: npt.NDArray[np.int32]) -> ArcImage:
    return ArcImage(img.tolist())


def logical_task(train_task_len: int) -> ArcTask:
    random_vals = RandomValues()
    in_imgs, out_imgs = logical_inout_task(
        train_task_len + 1,
        random_vals=random_vals,
    )
    in_arc_imgs = [np_img_to_arc_img(img) for img in in_imgs]
    out_arc_imgs = [np_img_to_arc_img(img) for img in out_imgs]

    inout_list = []
    for in_arc_img, out_arc_img in zip(in_arc_imgs, out_arc_imgs):
        inout_list.append(ArcInout(in_arc_img, out_arc_img))

    train_inout_list = inout_list[:train_task_len]
    test_inout_list = inout_list[-1]

    fake_candidate = in_arc_imgs
    #!dataclassesでOptionalが使えないのでとりあえずfakeとしてin_arc_imgsを使う
    rslt = ArcTask(train_inout_list, test_inout_list, fake_candidate)
    return rslt


FILE_EXTENTION = ".pkl"
def _save_logical_task(save_path: Path, task_len: int):
    task = logical_task(task_len)
    with save_path.open("wb") as f:
        pickle.dump(task, f)
    pickle.dump(task, save_path.open("wb"))
    return task


def _load_logical_task(f_path: Path) -> ArcTask:
    task = pickle.load(f_path.open("rb"))
    if not isinstance(task, ArcTask):
        raise ValueError(f"{f_path} is not ArcTask")
    return task


def test_save_and_load_logical_task(tmp_path: Path):
    f_path = tmp_path / f"test{FILE_EXTENTION}"
    saved_task= _save_logical_task(f_path, task_len=10)
    loaded_task = _load_logical_task(f_path)
    assert saved_task == loaded_task


def save_logical_tasks(save_dir: Path, task_num: int, task_len: int):
    print(f"save logical op tasks to {save_dir}...")

    if save_dir.exists():
        # all files in the directory will be deleted
        for file in save_dir.glob("*"):
            file.unlink()

    else:
        save_dir.mkdir(parents=True)

    # file_name_key_tasks = {}
    tasks = []
    for i in tqdm(range(task_num)):
        f_path = save_dir / f"{i:08}{FILE_EXTENTION}"
        task = _save_logical_task(f_path, task_len=task_len)
        tasks.append(task)

    return tasks


def load_logical_tasks(tasks_dir: Path) -> list[dict]:
    files = sorted(tasks_dir.glob(f"*{FILE_EXTENTION}"))
    tasks = []
    for file in files:
        task = _load_logical_task(file)
        tasks.append(
            {
                "input": str(task),
                "output": str(task.test_output),
            }
        )

    return tasks


def test_save_logical_tasks(tmp_path: Path):
    saved_tasks = save_logical_tasks(tmp_path, task_num=10, task_len=10)
    loaded_tasks = load_logical_tasks(tmp_path)

    for saved_task, loaded_task in zip(saved_tasks, loaded_tasks):
        assert loaded_task["input"] == str(saved_task)
        assert loaded_task["output"] == str(saved_task.test_output)

