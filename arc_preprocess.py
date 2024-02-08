# %%
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import json
import random
from typing import List, Union, Type, Optional
from arc_visualize import CH_source, MAX_SIZE

# VRTCL_DELIM = ","
VRTCL_DELIM = ""
HRZNTL_DELIM = "\n"
FAKE_NUM = 9
TRUE_NUM = 1
CANDIDATE_NUM = FAKE_NUM + TRUE_NUM
INOUT_CANDIDATE_NUM = int(CANDIDATE_NUM / 2)


class ArcImage:
    def __init__(self, original_2d_list: List[List[int]]) -> None:

        for row in original_2d_list:
            if len(row) != len(original_2d_list[0]):
                raise ValueError("All rows must have the same length.")
        if len(original_2d_list) > MAX_SIZE or len(original_2d_list[0]) > MAX_SIZE:
            raise ValueError("The size of the image is too large.")

        for row in original_2d_list:
            for num in row:
                if not isinstance(num, int):
                    raise ValueError("All elements must be int.")
                if num < 0:
                    raise ValueError("All elements must be > 0")
                if num > CH_source:
                    raise ValueError("All elements must be < {}".format(CH_source))

        self.img = original_2d_list

    def to_string(self, vr_delim, hr_delim) -> str:
        char_two_d_list = [[str(one_num) for one_num in row] for row in self.img]
        # num to char
        row_joined_list = [vr_delim.join(row) for row in char_two_d_list]
        return hr_delim.join(row_joined_list)

    def to_2d_list(self) -> List[List[int]]:
        return self.img

    def __str__(self) -> str:
        return self.to_string(vr_delim="", hr_delim="\n")


import pytest


class TestArcImage:

    def test_init_normal(self):
        test_image = [[i for i in range(11)] for _ in range(6)]
        arc_image = ArcImage(test_image)
        assert arc_image.img == test_image

    def test_init_lager_value(self):
        test_image = [[i for i in range(11)] for _ in range(6)]
        test_image[0][0] = 12
        with pytest.raises(ValueError):
            ArcImage(test_image)

    def test_init_smaller_value(self):
        test_image = [[i for i in range(11)] for _ in range(6)]
        test_image[0][0] = -1
        with pytest.raises(ValueError):
            ArcImage(test_image)

    def test_init_not_int(self):
        test_image = [[i for i in range(11)] for _ in range(6)]
        test_image[0][0] = "0"  # type: ignore
        with pytest.raises(ValueError):
            ArcImage(test_image)

    def test_init_not_same_length(self):
        test_image = [[i for i in range(11)] for _ in range(6)]
        test_image[0].append(1)
        with pytest.raises(ValueError):
            ArcImage(test_image)

    def test_to_string(self):
        test_image = [[i for i in range(11)] for _ in range(6)]
        arc_image = ArcImage(test_image)
        assert (
            arc_image.to_string(vr_delim=",", hr_delim="\n")
            == "0,1,2,3,4,5,6,7,8,9,10,11\n0,1,2,3,4,5,6,7,8,9,10,11\n0,1,2,3,4,5,6,7,8,9,10,11\n0,1,2,3,4,5,6,7,8,9,10,11\n0,1,2,3,4,5,6,7,8,9,10,11\n0,1,2,3,4,5,6,7,8,9,10,11"
        )

    def test_to_2d_list(self):
        test_image = [[i for i in range(11)] for _ in range(6)]
        arc_image = ArcImage(test_image)
        assert arc_image.to_2d_list() == test_image

    def test_str(self):
        test_image = [[1, 2], [3, 4]]
        arc_image = ArcImage(test_image)
        assert str(arc_image) == "12\n34"


if __name__ == "__main__":
    tst = TestArcImage()
    tst.test_init_normal()
    tst.test_init_lager_value()
    tst.test_init_smaller_value()
    tst.test_str()


# %%
@dataclass
class ArcInout:
    input: ArcImage
    output: ArcImage

    def __str__(self) -> str:
        return f"input:\n{self.input}\noutput:\n{self.output}"


if __name__ == "__main__":
    test_image = [[i for i in range(11)] for _ in range(6)]
    arc_image = ArcImage(test_image)
    arc_inout = ArcInout(arc_image, arc_image)
    print(arc_inout)

# %%
@dataclass
class ArcTask:
    train: list[ArcInout]
    test: ArcInout
    candidate: list[ArcImage]

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

    def to_str(self, train_name, test_name) -> str:
        rslt = ""

        for i, inout in enumerate(self.train):
            rslt += f"-{train_name}{i}-\n{inout}\n\n"

        rslt += f"-{test_name}-\n{self.test}\n\n"

        for i, one_candidate in enumerate(self.candidate):
            rslt += f"-candidate{i}-\n{one_candidate}\n"

        return rslt


    def __str__(self) -> str:
        return self.to_str("train", "test")



class TestArcTask:

    def test_str(self):
        test_image = [[1, 2], [3, 4]]
        arc_image = ArcImage(test_image)
        arc_inout = ArcInout(arc_image, arc_image)
        arc_task = ArcTask(
            train=[arc_inout, arc_inout],
            test=arc_inout,
            candidate=[arc_image],
        )
        need_str = "-train0-\ninput:\n12\n34\noutput:\n12\n34\n\n-train1-\ninput:\n12\n34\noutput:\n12\n34\n\n-test-\ninput:\n12\n34\noutput:\n12\n34\n\n-candidate0-\n12\n34\n"
        assert str(arc_task) == need_str


if __name__ == "__main__":
    tst = TestArcTask()
    tst.test_str()
# %%


def inout_to_string(inouts):
    inouts["input"] = ArcImage(inouts["input"])
    inouts["output"] = ArcImage(inouts["output"])
    return inouts


def train_n_test_to_string(train_n_test):
    train_n_test["train"] = {inout_to_string(inout) for inout in train_n_test["train"]}
    train_n_test["test"] = {inout_to_string(inout) for inout in train_n_test["test"]}
    return train_n_test

FAKE_NUM = 8
def task_json_to_arc_task(task):

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
    for two_cand in test[1:FAKE_NUM + 1]:
        candidate.append(ArcImage(two_cand["input"]))
        candidate.append(ArcImage(two_cand["output"]))
    
    return ArcTask(train, true_test_inout, candidate)

def path_to_arc_task(data_path: Path) -> List[ArcTask]:

    tasks = []
    for task_file in data_path.glob("*.json"):
        with task_file.open() as f:
            task = json.load(f)
            # print(task["name"])
            tasks.append(task_json_to_arc_task(task))

    return tasks

def test_path_to_arc_task():
    data_path = Path("data") / "training"
    tasks = path_to_arc_task(data_path)
    for task in tasks:
        print(task)

if __name__ == "__main__":
    test_path_to_arc_task()
#%%

def choice_test(original_question, choice_idx=None):
    # If there are multiple "test", leave one as it is and move the others to "train". choice_idx is the index of the "test" to leave as it is. For testing, it is set to a fixed value.

    question = deepcopy(original_question)
    if len(question["test"]) > INOUT_CANDIDATE_NUM:
        if choice_idx is None:
            choice_idx = (
                random.randint(0, len(question["test"]) - 1) // INOUT_CANDIDATE_NUM
            )
            # test len 以下のINOUT_CANDIDATE_NUMの倍数のidxを選ぶ

        keep_list = []
        for _ in range(INOUT_CANDIDATE_NUM):
            keep_list.append(question["test"].pop(choice_idx * INOUT_CANDIDATE_NUM))

        question["train"].extend(question["test"])
        question["test"] = keep_list

        question["choice_idx"] = choice_idx
        return question

    else:
        question["choice_idx"] = 0
        return question


def randamize_test(question):
    """
    本当のinputはtrue_inとして返す
    candidateは回答の選択肢をランダムに並び替えたもので
    ans_idx番目にtrue_outが入っている
    """
    true_inout = question["test"].pop(0)
    true_in = true_inout["input"]
    true_out = true_inout["output"]

    fake_outputs = []

    for inout in question["test"]:
        fake_outputs.append(inout["input"])
        fake_outputs.append(inout["output"])

    candidate = random.sample(fake_outputs, len(fake_outputs))

    ans_idx = random.randint(0, len(fake_outputs))
    candidate.insert(ans_idx, true_out)

    question["true_in"] = true_in
    question["true_out"] = true_out
    question["ans_idx"] = ans_idx
    question["candidate"] = candidate

    return question


def path_to_tasks(training_or_evaluation="training"):
    # data_path = Path("data") / training_or_evaluation
    tasks = []
    for task_file in data_path.glob("*.json"):
        with task_file.open() as f:
            task = json.load(f)
        task = train_n_test_to_string(task)
        task = choice_test(task)
        task = randamize_test(task)
        tasks.append(task)

    return tasks


def question_string_to_2d_list(question):
    for i, inout in enumerate(question["train"]):
        question["train"][i]["input"] = string_to_two_d_list(inout["input"])
        question["train"][i]["output"] = string_to_two_d_list(inout["output"])

    for i, inout in enumerate(question["test"]):
        question["test"][i]["input"] = string_to_two_d_list(inout["input"])
        question["test"][i]["output"] = string_to_two_d_list(inout["output"])

    question["true_in"] = string_to_two_d_list(question["true_in"])
    question["true_out"] = string_to_two_d_list(question["true_out"])

    #! candidateにNoneがでてエラー
    # for cand in question["candidate"]:
    #     print(cand)

    # question["candidate"] = [string_to_two_d_list(cand) for cand in question["candidate"]]
    return question


def make_example_prompt(train_idx, inout):
    return f"example {train_idx}\n{inout['input']}\n->\n{inout['output']}\n\n"


def make_prompt(question):
    # prompt = "Instract: \n"
    prompt = ""
    prompt += "".join(
        make_example_prompt(train_idx, inout)
        for train_idx, inout in enumerate(question["train"])
    )
    prompt += "question\n" + question["true_in"] + "\n->\n"
    # prompt += "Output: "
    return prompt


def test_choice_test():
    MAX_CHOICE_NUM = 3
    TEST_NUM = 5

    for choice_idx in range(MAX_CHOICE_NUM):
        origin_question = {
            "train": [-5 + i for i in range(5)],
            "test": [i for i in range(MAX_CHOICE_NUM * TEST_NUM)],
        }
        question = choice_test(origin_question, choice_idx)
        assert question["choice_idx"] == choice_idx
        assert (
            question["test"]
            == origin_question["test"][
                choice_idx * TEST_NUM : (choice_idx + 1) * TEST_NUM
            ]
        )

        moved_task = set(origin_question["test"]) - set(question["test"])
        assert moved_task <= set(question["train"])

    question = {"train": [-5 + i for i in range(5)], "test": [i for i in range(5)]}

    choided_flag = {i: False for i in range(MAX_CHOICE_NUM)}

    # ランダムに選ばれているかを確認
    for _ in range(100):
        origin_question = {
            "train": [-5 + i for i in range(5)],
            "test": [i for i in range(MAX_CHOICE_NUM * TEST_NUM)],
        }
        question = choice_test(origin_question)
        if question["choice_idx"] in choided_flag.keys():
            choided_flag[question["choice_idx"]] = True
    assert all(choided_flag.values())


test_choice_test()


def test_randamize_test():
    question = {
        "train": [],
        "test": [
            {"input": i * 2, "output": i * 2 + 1} for i in range(INOUT_CANDIDATE_NUM)
        ],
    }
    print(question)
    question = randamize_test(question)
    print(question["true_in"])
    print(question["ans_idx"])
    print(question["candidate"])
    assert question["true_in"] == 0
    assert question["candidate"][question["ans_idx"]] == 1

    ans_idx_flag = {i: 0 for i in range(CANDIDATE_NUM - 1)}
    test_num = 100

    for _ in range(test_num * CANDIDATE_NUM):
        question = {
            "train": [],
            "test": [
                {"input": i * 2, "output": i * 2 + 1}
                for i in range(INOUT_CANDIDATE_NUM)
            ],
        }
        question = randamize_test(question)
        ans_idx_flag[question["ans_idx"]] += 1

    print(ans_idx_flag)
    assert all(ans_idx_flag.values())
    assert question["true_in"] == 0
    assert question["candidate"][question["ans_idx"]] == 1


from arc_visualize import plot_task


def test_make_2d_list_to_string():
    questions = make_2d_list_to_string("training")
    for question in questions:
        question = question_string_to_2d_list(question)
        if len(question["train"]) > 5:
            print(question["name"])
            print(question["train"])
            plot_task(
                question["train"],
                question["true_in"],
                question["true_out"],
                # candidate=question["candidate"],
                # model_answer=question["candidate"],
                fold=10,
            )


test_make_2d_list_to_string()
# %%
