import joblib
from pathlib import Path
from typing import Any,Tuple, Optional,  Union, Callable, Dict, Iterable, List
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import torch
from .mono_mnist import MonoMNIST


class CollectionMonoMNIST():
    """A collection of all monochrome MNIST datasets (ie. Red, Green, Blue, Gray)
    generated with the same seed and the split pkl data from the data_dir
    """
    def __init__(self, *,
                 data_root: Path,
                 seed:int,
                 train:bool=True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.data_root = data_root
        self.seed = seed
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self._create_collection() # creates each mono-MNIST as an attribute


    def _create_collection(self) -> None:
        """
        Creates each mono-MNIST and sets them as an attribute
        """
        breakpoint()
        self.GrayMNIST = MonoMNIST(data_root=self.data_root,
                              color='gray',
                              transform=self.transform,
                              target_transform=self.target_transform,
                              train=self.train)

        self.RedMNIST = MonoMNIST(data_root=self.data_root,
                             color='red',
                             transform=self.transform,
                             target_transform=self.target_transform,
                             train=self.train)
        self.GreenMNIST = MonoMNIST(data_root=self.data_root,
                               color='green',
                               transform=self.transform,
                               target_transform=self.target_transform,
                               train=self.train)
        self.BlueMNIST = MonoMNIST(data_root=self.data_root,
                              color='blue',
                              transform=self.transform,
                              target_transform=self.target_transform,
                              train=self.train)

    def get_mono_dsets(self, colors: List[str]) -> List[MonoMNIST]:
        dsets = []
        for c in colors:
            dsets.append(getattr(self, f"{c.lower().capitalize()}MNIST"))
        return dsets

class MultiMonoMNIST(Dataset):
    """
    - data_root: root dir that contains "mnist_{color}.pkl" files
    - color: str; one of "red", "green", "blue"
    - transform
    - target_transform
    - download
    - seed
    - use_train_dataset

    Eg.
    - data_root: Path('/data/hayley-old/Tenanbaum2000/data/Mono-MNIST/')
        - mnist_data_root: data_root.parent

    """
    _name_formatspec = "MNIST-{colors_str}_seed-{seed}"

    def __init__(self,
            data_root: Union[Path, str],
            colors: List[str],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            seed: int=123,
            train: bool=True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.colors = [c.lower() for c in colors]
        for c in self.colors:
            assert c in ["gray", "red", "green", "blue"], "color must be one of gray, red, green, blue"

        # Extra transform that will be applied after the base transforms, ToTensor() and Monochromizer()
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed
        self.train = train
        self.mode = 'train' if self.train else 'test'

        # Create a concatenated dataset from multiple MonoMNIST datasets as
        # specified in the input argument `colors`
        self.dsets = self.get_mono_dsets(self.colors)
        self.ds = ConcatDataset(self.dsets)

    def get_mono_dsets(self, colors: List[str]) -> List[MonoMNIST]:
        dsets = []
        for color in colors:
            ds = MonoMNIST(data_root=self.data_root,
                              color=color,
                              transform=self.transform,
                              target_transform=self.target_transform,
                              train=self.train)
            dsets.append(ds)
        return dsets

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index:int) -> Tuple[Any,Any]:
        return self.ds[index]

    @property
    def name(self) -> str:
        return self._name_formatspec.format(
            colors_str='-'.join(self.colors),
            seed=self.seed
        )
