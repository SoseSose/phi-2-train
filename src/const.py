from dataclasses import dataclass

@dataclass
class ArcConst:
    COLOR :list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    MIN_COLOR_NUM = min(COLOR)
    MAX_COLOR_NUM = max(COLOR)
    PAD_COLOR:int = 10 # 10はPADの色
    MAX_IMG_SIZE:int = 30
    MAX_TRAIN_SIZE:int = 10
    MIN_IMG_SIZE:int = 1

    FAKE_NUM = 8#FAKEありのデータセットを作ったので,1つのタスクに含まれるFAKEの数


