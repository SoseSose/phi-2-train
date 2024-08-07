# %%
import json
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import List, Union

import numpy as np
import numpy.typing as npt
import pytest

from data_processing.const import ArcConst

MIN_COLOR_NUM = ArcConst().MIN_COLOR_NUM
MAX_COLOR_NUM = ArcConst().MAX_COLOR_NUM
MAX_SIZE = ArcConst().MAX_IMG_SIZE
VR_DELIM = ""
HR_DELIM = "\n"

class ArcImage:
    def __init__(
        self, original_2d_list: Union[List[List[int]], npt.NDArray[np.int32]]
    ) -> None:
        for row in original_2d_list:
            if len(row) != len(original_2d_list[0]):
                raise ValueError("All rows must have the same length.")
        if len(original_2d_list) > MAX_SIZE or len(original_2d_list[0]) > MAX_SIZE:
            raise ValueError("The size of the image is too large.")

        for row in original_2d_list:
            for num in row:
                if not isinstance(num, int):
                    raise ValueError("All elements must be int.")
                if num < MIN_COLOR_NUM:
                    raise ValueError("All elements must be > 0")
                if num > MAX_COLOR_NUM:
                    raise ValueError("All elements must be < {}".format(MAX_COLOR_NUM))

        if isinstance(original_2d_list, list):
            self.img = original_2d_list
        else:
            self.img = original_2d_list.tolist()

        self.x = len(original_2d_list[0])
        self.y = len(original_2d_list)

    def to_str(self) -> str:
        char_two_d_list = [[str(one_num) for one_num in row] for row in self.img]
        # num to char
        row_joined_list = [VR_DELIM.join(row) for row in char_two_d_list]
        return HR_DELIM.join(row_joined_list)

    def to_2d_list(self) -> List[List[int]]:
        return self.img

    def __str__(self) -> str:
        return self.to_str()

    @property
    def to_np(self) -> npt.NDArray[np.int32]:
        return np.array(self.img)

    def __array__(self):
        return np.array(self.img)

    def __eq__(self, other: "ArcImage") -> bool:
        if self.to_np.shape != other.to_np.shape:
            return False
        else:
            return (self.to_np == other.to_np).all()


class TestArcImage:
    def get_test_image(self) -> List[List[int]]:
        return [[i for i in range(ArcConst().MIN_COLOR_NUM,ArcConst().MAX_COLOR_NUM+1)] for _ in range(6)]
    def test_init_normal(self):
        test_image = self.get_test_image()
        arc_image = ArcImage(test_image)
        assert arc_image.img == test_image

    def test_init_lager_value(self):
        test_image = self.get_test_image()
        test_image[0][0] = 12
        with pytest.raises(ValueError):
            ArcImage(test_image)

    def test_init_smaller_value(self):
        test_image = self.get_test_image()
        test_image[0][0] = -1
        with pytest.raises(ValueError):
            ArcImage(test_image)

    def test_init_not_int(self):
        test_image = self.get_test_image()
        test_image[0][0] = "0"  # type: ignore
        with pytest.raises(ValueError):
            ArcImage(test_image)

    def test_init_not_same_length(self):
        test_image = self.get_test_image()
        test_image[0].append(1)
        with pytest.raises(ValueError):
            ArcImage(test_image)

    def test_to_string(self):
        test_image = self.get_test_image()
        arc_image = ArcImage(test_image)
        assert (
            arc_image.to_str()
            == "0123456789\n0123456789\n0123456789\n0123456789\n0123456789\n0123456789"
        )

    def test_to_2d_list(self):
        test_image = self.get_test_image()
        arc_image = ArcImage(test_image)
        assert arc_image.to_2d_list() == test_image

    def test_str(self):
        test_image = [[1, 2], [3, 4]]
        arc_image = ArcImage(test_image)
        assert str(arc_image) == "12\n34"
    
    def test_ArcImage_equivarent_return_True(self):
        test_image = [[1, 2], [3, 4]]
        arc_image = ArcImage(test_image)
        arc_image2 = ArcImage(test_image)
        assert arc_image == arc_image2
    
    def test_ArcImage_unequivarent_return_False(self):
        arc_image = ArcImage([[1, 2], [3, 4]])
        arc_image2 = ArcImage([[1, 2], [3, 5]])
        assert arc_image != arc_image2

    def test_ArcImage_eq_not_same_shape_return_False(self):
        arc_image = ArcImage([[1, 2], [3, 4]])
        arc_image2 = ArcImage([[1, 2, 2], [3, 4,4]])
        assert arc_image != arc_image2


@dataclass
class ArcInout:
    input: ArcImage
    output: ArcImage

    def __str__(self) -> str:
        # return f"input:\n{self.input}\noutput:\n{self.output}"
        return f"{self.input}\n->\n{self.output}"

    def __eq__(self, other: "ArcInout") -> bool:
        return self.input == other.input and self.output == other.output

def test_Arc_inout_input_not_same_return_false():
    test_image = [[1,2], [3,4]]
    arc_inout = ArcInout(ArcImage(test_image), ArcImage(test_image))
    arc_inout2 = ArcInout(ArcImage([[1,2],[3, 5]]), ArcImage(test_image))

    assert arc_inout != arc_inout2

def test_Arc_inout_output_not_same_return_false():
    test_image = [[1,2], [3,4]]
    arc_inout = ArcInout(ArcImage(test_image), ArcImage(test_image))
    arc_inout2 = ArcInout(ArcImage([[1,2],[3, 5]]), ArcImage(test_image))

    assert arc_inout != arc_inout2

def test_Arc_inout_input_same_return_true():
    test_image = [[1,2], [3,4]]
    arc_inout = ArcInout(ArcImage(test_image), ArcImage(test_image))
    arc_inout2 = ArcInout(ArcImage(test_image), ArcImage(test_image))

    assert arc_inout == arc_inout2

TRAIN_NAME = "train"
TEST_NAME = "test"

@dataclass
class ArcTask:
    train: list[ArcInout]
    test: ArcInout
    candidate: list[ArcImage]
    name: str = "no name"

    @property
    def train_inputs(self) -> list[ArcImage]:
        return [inout.input for inout in self.train]

    @property
    def train_outputs(self) -> list[ArcImage]:
        return [inout.output for inout in self.train]

    @property
    def test_input(self) -> Union[ArcImage, list[ArcImage]]:
        if isinstance(self.test, list):
            return [inout.input for inout in self.test]
        else:
            return self.test.input

    @property
    def test_output(self) -> Union[ArcImage, list[ArcImage]]:
        if isinstance(self.test, list):
            return [inout.output for inout in self.test]
        else:
            return self.test.output

    def to_str(
        self,
        show_test_out: bool = False,
        show_candidata: bool = False,
    ) -> str:
        rslt = ""

        for i, inout in enumerate(self.train):
            rslt += f"-{TRAIN_NAME}{i}-\n{inout}\n\n"

        if show_candidata:
            for i, one_candidate in enumerate(self.candidate):
                rslt += f"-candidate{i}-\n{one_candidate}\n\n"

        rslt += f"-{TEST_NAME}-\n"
        rslt += f"{self.test_input}\n->"

        if show_test_out:
            rslt += f"\n{self.test_output}"

        return rslt
    
    @property
    def question(self) -> str:
        return self.to_str()
    
    def __str__(self) -> str:
        return self.to_str()

    def __eq__(self, other: "ArcTask") -> bool:
        
        return (
            self.train == other.train
            and self.test == other.test
            and self.candidate == other.candidate
        )

class TestArcTask:
    def test_train_inputs(self):
        true_arc_image = ArcImage([[1, 2], [3, 4]])
        false_arc_image = ArcImage([[5, 6], [7, 8]])
        arc_inout = ArcInout(true_arc_image, false_arc_image)
        arc_task = ArcTask(
            train=[arc_inout, arc_inout],
            test=arc_inout,
            candidate=[false_arc_image],
        )
        assert arc_task.train_inputs == [true_arc_image, true_arc_image]

    def test_train_outputs(self):
        true_arc_image = ArcImage([[1, 2], [3, 4]])
        false_arc_image = ArcImage([[5, 6], [7, 8]])
        arc_inout = ArcInout(true_arc_image, false_arc_image)
        arc_task = ArcTask(
            train=[arc_inout, arc_inout],
            test=arc_inout,
            candidate=[false_arc_image],
        )
        assert arc_task.train_outputs == [false_arc_image, false_arc_image]

    def test_test_input(self):
        true_arc_image = ArcImage([[1, 2], [3, 4]])
        false_arc_image = ArcImage([[5, 6], [7, 8]])
        arc_inout = ArcInout(true_arc_image, false_arc_image)
        arc_task = ArcTask(
            train=[arc_inout, arc_inout],
            test=arc_inout,
            candidate=[false_arc_image],
        )
        assert arc_task.test_input == true_arc_image

    def test_test_output(self):
        true_arc_image = ArcImage([[1, 2], [3, 4]])
        false_arc_image = ArcImage([[5, 6], [7, 8]])
        arc_inout = ArcInout(true_arc_image, false_arc_image)
        arc_task = ArcTask(
            train=[arc_inout, arc_inout],
            test=arc_inout,
            candidate=[false_arc_image],
        )
        assert arc_task.test_output == false_arc_image

    def test_str(self):
        test_image = [[1, 2], [3, 4]]
        arc_image = ArcImage(test_image)
        arc_inout = ArcInout(arc_image, arc_image)
        arc_task = ArcTask(
            train=[arc_inout, arc_inout],
            test=arc_inout,
            candidate=[arc_image],
        )

        rslt = arc_task.to_str(show_candidata=True, show_test_out=True)

        need_str = """\
        -train0-
        12
        34
        ->
        12
        34
        
        -train1-
        12
        34
        ->
        12
        34

        -candidate0-
        12
        34
        
        -test-
        12
        34
        ->
        12
        34
        """

        need_str = textwrap.dedent(need_str).strip()
        assert rslt == need_str


def test_ArcTask_eq_same_return_True():
    test_image = [[1, 2], [3, 4]]
    arc_image = ArcImage(test_image)
    arc_inout = ArcInout(arc_image, arc_image)
    arc_task = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout,
        candidate=[arc_image],
    )
    arc_task2 = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout,
        candidate=[arc_image],
    )
    assert arc_task == arc_task2


def test_ArcTask_eq_not_same_train_return_False():
    test_image = [[1, 2], [3, 4]]
    arc_image = ArcImage(test_image)
    arc_inout = ArcInout(arc_image, arc_image)
    arc_inout2 = ArcInout(arc_image, ArcImage([[1, 2], [3, 5]]))
    arc_task = ArcTask(
        train=[arc_inout, arc_inout2],
        test=arc_inout,
        candidate=[arc_image],
    )
    arc_task2 = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout,
        candidate=[arc_image],
    )
    assert arc_task != arc_task2

def test_ArcTask_eq_not_same_test_return_False():
    test_image = [[1, 2], [3, 4]]
    arc_image = ArcImage(test_image)
    arc_inout = ArcInout(arc_image, arc_image)
    arc_inout2 = ArcInout(arc_image, ArcImage([[1, 2], [3, 5]]))
    arc_task = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout,
        candidate=[arc_image],
    )
    arc_task2 = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout2,
        candidate=[ArcImage([[1, 2], [3, 5]])],
    )
    assert arc_task != arc_task2

def test_ArcTask_eq_not_same_candidate_return_False():
    test_image = [[1, 2], [3, 4]]
    arc_image = ArcImage(test_image)
    arc_inout = ArcInout(arc_image, arc_image)
    arc_task = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout,
        candidate=[arc_image],
    )
    arc_task2 = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout,
        candidate=[ArcImage([[1, 2], [3, 5]])],
    )
    assert arc_task != arc_task2

class ArcTaskSet:
    """
    Represents a set of ARC tasks.

    Methods:
    - task_json_to_arc_task(task): Converts a task in JSON format to an ArcTask object.
    - path_to_arc_task(data_path): Converts a path to a directory containing JSON task files to a list of ArcTask objects.
    """

    def _task_json_to_arc_task(self, task, name: str):
        """
        Converts a task in JSON format to an ArcTask object.

        Parameters:
        - task: A dictionary representing a task in JSON format.

        Returns:
        - An ArcTask object representing the converted task.
        """

        train = []
        for inout in task["train"]:
            input = ArcImage(inout["input"])
            output = ArcImage(inout["output"])
            train.append(ArcInout(input, output))

        test = task["test"]
        true_test_in = test[0]["input"]
        true_test_out = test[0]["output"]

        true_test_inout = ArcInout(ArcImage(true_test_in), ArcImage(true_test_out))

        candidate = []
        for two_cand in test[1 : ArcConst().FAKE_NUM + 1]:
            candidate.append(ArcImage(two_cand["input"]))
            candidate.append(ArcImage(two_cand["output"]))

        return ArcTask(train, true_test_inout, candidate, name)

    def path_to_arc_task(self, data_path: str) -> List[ArcTask]:
        """
        Converts a path to a directory containing JSON task files to a list of ArcTask objects.

        Parameters:
        - data_path: A Path object representing the path to the directory containing JSON task files.

        Returns:
        - A list of ArcTask objects representing the converted tasks.
        """
        tasks = []
        for task_file in Path(data_path).glob("*.json"):
            with task_file.open() as f:
                task = json.load(f)
                tasks.append(self._task_json_to_arc_task(task, task["name"]))

        return tasks


def str_to_arc_image(string: str) -> ArcImage:
    """
    Converts a string representation of a 2D list to a 2D list.

    Parameters:
    - string: A string representation of a 2D list.

    Returns:
    - A 2D list representing the converted 2D list.
    """
    two_d_list = string.split("\n")
    two_d_list = [list(row) for row in two_d_list]
    two_d_list = [[int(num) for num in row] for row in two_d_list]
    two_d_list = ArcImage(two_d_list)
    return two_d_list


class TestArcTaskSet:
    def test_op_path_to_arc_task(self):
        arc_task_set = ArcTaskSet()
        _ = arc_task_set.path_to_arc_task("data/training")
