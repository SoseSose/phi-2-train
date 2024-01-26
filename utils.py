#%%
import random
import numpy as np
import torch

def fix_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    print("random seed fixed to {}. Warn !! It is Global".format(seed))
    #! ライブラリ側でテストされていると考えて個別にテストしない

