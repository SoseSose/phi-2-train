import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class ArcConst:
    # COLOR: tuple[int] = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    MIN_COLOR_NUM: int = 0
    MAX_COLOR_NUM: int = 9
    PAD_COLOR: int = 10  # 10はPADの色
    MAX_TRAIN_SIZE: int = 10
    MAX_IMG_SIZE: int = 30
    MIN_IMG_SIZE: int = 1

    FAKE_NUM = 8  # FAKEありのデータセットを作ったので,1つのタスクに含まれるFAKEの数


class Color:
    def __init__(self):
        self.color_cand = [
            i for i in range(ArcConst.MIN_COLOR_NUM, ArcConst.MAX_COLOR_NUM)
        ]

    def pick_random_unused(self, index: Optional[int] = None) -> int:
        if index is not None:
            if index < 0 or index >= len(self.color_cand):
                raise Exception(f"index must be 0 <= index < {len(self.color_cand)}")
            picked = self.color_cand.pop(index)
        else:
            picked = self.color_cand.pop(random.randint(0, len(self.color_cand) - 1))
        return picked
