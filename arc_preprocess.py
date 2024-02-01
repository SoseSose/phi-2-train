# %%
from copy import deepcopy
from pathlib import Path
import json
import random

# VRTCL_DELIM = ","
VRTCL_DELIM = ""
HRZNTL_DELIM = "\n"
FAKE_NUM = 9
TRUE_NUM = 1
CANDIDATE_NUM = FAKE_NUM + TRUE_NUM
INOUT_CANDIDATE_NUM = int(CANDIDATE_NUM / 2)


def two_d_list_to_string(list_2d):
    char_two_d_list = [[str(one_num) for one_num in row] for row in list_2d]
    # num to char
    row_joined_list = [VRTCL_DELIM.join(row) for row in char_two_d_list]
    return HRZNTL_DELIM.join(row_joined_list)


def string_to_two_d_list(strings):
    rows = strings.split(HRZNTL_DELIM)
    delim_removed_rows = [row.replace(VRTCL_DELIM, "") for row in rows]

    return [[int(one_num) for one_num in row] for row in delim_removed_rows]


def inout_to_string(inouts):
    inouts["input"] = two_d_list_to_string(inouts["input"])
    inouts["output"] = two_d_list_to_string(inouts["output"])
    return inouts


def train_n_test_to_string(train_n_test):
    train_n_test["train"] = [inout_to_string(inout) for inout in train_n_test["train"]]
    train_n_test["test"] = [inout_to_string(inout) for inout in train_n_test["test"]]
    return train_n_test


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


def make_2d_list_to_string(training_or_evaluation="training"):
    data_path = Path("data") / training_or_evaluation
    questions = []
    for task_file in data_path.glob("*.json"):
        with task_file.open() as f:
            question = json.load(f)
        question = train_n_test_to_string(question)
        question = choice_test(question)
        question = randamize_test(question)
        questions.append(question)

    return questions


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