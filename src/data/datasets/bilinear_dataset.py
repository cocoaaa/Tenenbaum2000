import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Set, Any, Dict
from torch.utils.data import Dataset

class BilinearDataset(Dataset):
    def __getitem__(self, item:int) -> Dict[str, Any]:
        """Must return a dict that has required key,value pairs:
        - "x": (torch.Tensor) of a single datapoint
        - "content"

        Required keys:
        "x", "content", "style"
        - dict[x] ->
        raise NotImplementedError

