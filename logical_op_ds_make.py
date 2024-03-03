# %%
import random
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
    from_img: np.ndarray, to_img: np.ndarray, point_x: int, point_y: int
) -> np.ndarray:
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

    from_img = ArcImage([[1 for _ in range(MAX_SIZE)] for _ in range(MAX_SIZE)])
    to_img = ArcImage([[0 for _ in range(2)] for _ in range(2)])
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
    from_img: np.ndarray,
    x_from: int,
    x_to: int,
    y_from: int,
    y_to: int,
    fill_val: int,
):

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
    input = np.zeros((5, 5))
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
    def __init__(self):
        self.a_0_b_0 = random.randint(0, 1)
        self.a_0_b_1 = random.randint(0, 1)
        self.a_1_b_0 = random.randint(0, 1)
        self.a_1_b_1 = random.randint(0, 1)

    def calc(self, a: npt.NDArray[np.int32], b: npt.NDArray[np.int32]):

        if a.shape != b.shape:
            raise ValueError("a and b must have the same shape")

        if ((a != 0) & (a != 1)).all():
            raise ValueError("a should be 0 or 1")

        if ((b != 0) & (b != 1)).all():
            raise ValueError("b should be 0 or 1")

        return np.where(
            (a == 0) & (b == 0),
            self.a_0_b_0,
            np.where(
                (a == 0) & (b == 1),
                self.a_0_b_1,
                np.where((a == 1) & (b == 0), self.a_1_b_0, self.a_1_b_1),
            ),
        )


def test_RandomLogicalOp():

    # check calc output is 0 or 1
    for _ in range(100):
        random_logical_op = RandomLogicalOp()

        a = np.array([[0 for _ in range(2)] for _ in range(2)])
        b = np.array([[0 for _ in range(2)] for _ in range(2)])
        rslt = random_logical_op.calc(a, b)
        assert ((rslt == 0) | (rslt == 1)).all()

        a = np.array([[0 for _ in range(2)] for _ in range(2)])
        b = np.array([[1 for _ in range(2)] for _ in range(2)])
        rslt = random_logical_op.calc(a, b)
        assert ((rslt == 0) | (rslt == 1)).all()

        a = np.array([[1 for _ in range(2)] for _ in range(2)])
        b = np.array([[0 for _ in range(2)] for _ in range(2)])
        rslt = random_logical_op.calc(a, b)
        assert ((rslt == 0) | (rslt == 1)).all()

        a = np.array([[1 for _ in range(2)] for _ in range(2)])
        b = np.array([[1 for _ in range(2)] for _ in range(2)])
        rslt = random_logical_op.calc(a, b)
        assert ((rslt == 0) | (rslt == 1)).all()

    # case a is not 0 or 1
    a = np.array([[2 for _ in range(2)] for _ in range(2)])
    b = np.array([[0 for _ in range(2)] for _ in range(2)])
    with pytest.raises(Exception) as e:
        random_logical_op.calc(a, b)
    assert str(e.value) == "a should be 0 or 1"

    # case b is not 0 or 1
    a = np.array([[0 for _ in range(2)] for _ in range(2)])
    b = np.array([[2 for _ in range(2)] for _ in range(2)])
    with pytest.raises(Exception) as e:
        random_logical_op.calc(a, b)
    assert str(e.value) == "b should be 0 or 1"


MAX_PART_IMG_SIZE = int(MAX_SIZE / 2) - 1


def two_img_concat_with_line(
    img1: np.ndarray,
    img2: np.ndarray,
    line_color: int,
) -> np.ndarray:
    # img1とimg2をline_colorで区切った画像を生成する。
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must have the same shape")

    if img1.shape[0] > MAX_PART_IMG_SIZE or img1.shape[1] > MAX_PART_IMG_SIZE:
        raise ValueError(
            f"img1 and img2 must be smaller than {MAX_PART_IMG_SIZE + 1}(MAX_PART_IMG_SIZE)"
        )
    img1_y, img1_x = img1.shape[:2]

    rslt_img = np.full((img1_y * 2 + 1, img1_x), line_color, dtype=int)
    rslt_img = paste(rslt_img, img1, 0, 0)
    rslt_img = paste(rslt_img, img2, 0, img1_y + 1)

    return rslt_img


def test_two_img_concat_with_line():
    img1 = np.array([[1, 1], [1, 1]])
    img2 = np.array([[0, 0], [0, 0]])
    rslt = two_img_concat_with_line(img1, img2, 8)
    print(rslt)


def test_img1_and_img2_must_have_the_same_shape():
    img1 = np.array([[1, 1], [1, 1]])
    img2 = np.array([[0, 0, 0], [0, 0, 0]])
    with pytest.raises(Exception) as e:
        rslt = two_img_concat_with_line(img1, img2, 8)
    assert str(e.value) == "img1 and img2 must have the same shape"


def test_img1_and_img2_must_be_smaller_than_MAX_PART_IMG_SIZE():
    big_img = [
        [1 for _ in range(MAX_PART_IMG_SIZE)] for _ in range(MAX_PART_IMG_SIZE + 1)
    ]

    img1 = np.array(big_img)
    img2 = np.array(big_img)
    with pytest.raises(Exception) as e:
        rslt = two_img_concat_with_line(img1, img2, 8)
    assert (
        str(e.value)
        == f"img1 and img2 must be smaller than {MAX_PART_IMG_SIZE+1}(MAX_PART_IMG_SIZE)"
    )

    big_img = [
        [1 for _ in range(MAX_PART_IMG_SIZE + 1)] for _ in range(MAX_PART_IMG_SIZE)
    ]

    img1 = np.array(big_img)
    img2 = np.array(big_img)
    with pytest.raises(Exception) as e:
        rslt = two_img_concat_with_line(img1, img2, 8)
    assert (
        str(e.value)
        == f"img1 and img2 must be smaller than {MAX_PART_IMG_SIZE+1}(MAX_PART_IMG_SIZE)"
    )


def make_random_box(size_x: int, size_y: int, cand_val: list[int]):
    return np.random.choice(cand_val, size=(size_y, size_x))


def test_make_random_box():
    rslt = make_random_box(2, 2, [0, 1])
    assert rslt.shape == (2, 2)
    assert ((rslt == 0) | (rslt == 1)).all()


class Color:
    def __init__(self):
        self.color_cand = [i for i in range(CH_source)]

    def pick_random_unused(self, index: Optional[int] = None):
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

    def to_binary(self, img: np.ndarray):

        if ((img != self.zero_color) & (img != self.one_color)).all():
            raise ValueError(f"img val should be {self.zero_color} or {self.one_color}")

        return np.where(img == self.zero_color, 0, 1)

    def to_color(self, img: np.ndarray):
        if ((img != 0) & (img != 1)).all():
            raise ValueError("img val should be 0 or 1")

        return np.where(img == 0, self.zero_color, self.one_color)


def logical_out_img(
    img1: np.ndarray,
    img2: np.ndarray,
    color_converter: ColorConverter,
    logical_op: RandomLogicalOp,
):
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must have the same shape")

    img1_binary = color_converter.to_binary(img1)
    img2_binary = color_converter.to_binary(img1)

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
    logical_op = RandomLogicalOp()
    rslt = logical_out_img(img1, img2, color_converter, logical_op)
    assert rslt.shape == (2, 2)
    assert ((rslt == zero_color) | (rslt == one_color)).all()

    zero_color = 3
    one_color = 8
    img1 = np.array([[zero_color, one_color], [zero_color, one_color]])
    img2 = np.array([[zero_color], [one_color]])
    with pytest.raises(Exception) as e:
        rslt = logical_out_img(img1, img2, color_converter, logical_op)
    assert str(e.value) == "img1 and img2 must have the same shape"


def logical_inout_img(
    logical_op: RandomLogicalOp,
    zero_color: int,
    one_color: int,
    line_color: int,
    is_vertical: bool,
):

    size_x = random.randint(1, MAX_PART_IMG_SIZE)
    size_y = random.randint(1, MAX_PART_IMG_SIZE)

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

    img1 = in_img[:part_img_size+1]
    line = in_img[part_img_size + 1]
    img2 = in_img[part_img_size + 2 :]

    # img1, img2 shold be zero_color or one_color
    assert ((img1 == zero_color) | (img1 == one_color)).all()
    assert ((img2 == zero_color) | (img2 == one_color)).all()
    # line should be line_color
    assert (line == line_color).all()
    # out_img should be zero_color or one_color
    assert ((out_img == zero_color) | (out_img == one_color)).all()


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
    in_img_x = in_img.shape[1]
    part_img_size = int(in_img_x / 2) - 1

    img1 = in_img[:, :part_img_size + 1]
    line = in_img[:, part_img_size + 1]
    img2 = in_img[:, part_img_size + 2 :]

    # img1, img2 shold be zero_color or one_color
    assert ((img1 == zero_color) | (img1 == one_color)).all()
    assert ((img2 == zero_color) | (img2 == one_color)).all()
    # line should be line_color
    assert (line == line_color).all()
    # out_img should be zero_color or one_color
    assert ((out_img == zero_color) | (out_img == one_color)).all()


def logical_inout_imgs(img_num: int):
    logical_op = RandomLogicalOp()
    color = Color()
    zero_color = color.pick_random_unused()
    one_color = color.pick_random_unused()
    line_color = color.pick_random_unused()
    is_vertical = random.randint(0, 1) == 0
    in_imgs = []
    out_imgs = []

    for _ in range(img_num):
        in_img, out_img = logical_inout_img(
            logical_op, zero_color, one_color, line_color, is_vertical
        )
        in_imgs.append(in_img)
        out_imgs.append(out_img)

    return in_imgs, out_imgs


in_imgs, out_imgs = logical_inout_imgs(10)
for in_img, out_img in zip(in_imgs, out_imgs):
    print("in")
    print(in_img)
    print("out")
    print(out_img)
