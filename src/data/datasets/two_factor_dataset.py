"""
Base Dataset class for datasets with two kinds of labels, "content label" and "style label".
The content label is compatiable with the target label in a standard dataset for a supervised learning problem.
The style label is compatible with the domain label in a standard dataset for a domain-adaptation problem.

A two-factor dataset returns an item as a dictionary with keys:
- "x" (required): torch.Tensor
- "<name_of_content_class>": eg. "digit" for Colorized MNIST dataset
- "<name_of_style_class>" : eg. "color" for Colorized MNIST dataset

Any subclass must implement:
- self.__getitem__(idx)
- self.unpack(batch) -> Tuple of x,y,d, where
    x is the main data input (eg. an image),
    y is the content-label, and
    d is the domain/style label
- self.keys()
Examples
--------


"""
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Set, Any, Dict, Tuple
from torch.utils.data import Dataset

class TwoFactorDataset(Dataset):

    _keys = None # Required : List[str]

    def __init__(self):
        super().__init__()

    def __getitem__(self, item:int) -> Dict[str, Any]:
        """Must return a dict that has required key,value pairs:
        - "x": (torch.Tensor) of a single datapoint; Required
        - "<name_of_content_class>": eg. "digit"
        - "<name_of_style_class>": eg. "color" for MonoMNIST dataset or, "style" for Maptiles dataset
        """
        raise NotImplementedError

    @classmethod
    def keys(cls) -> List[str]:
        """Returns a list of keys of an item (which is a dictionary) in the dataset
        """
        # raise NotImplementedError
        return cls._keys

    @classmethod
    def unpack(cls, batch: Dict[str, Any]) -> Tuple[Any]:
        return [batch[k] for k in cls.keys()]



