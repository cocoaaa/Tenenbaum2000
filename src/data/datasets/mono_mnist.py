import joblib
from pathlib import Path
from typing import Any,Tuple, Optional,  Union, Callable, Dict, Iterable, List
from collections import defaultdict
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from src.data.transforms.transforms import  Monochromizer


class MonoMNIST(Dataset):
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
    _fn_formatspec = "{mode}_mnist_{color}_seed-{seed}.pkl"
    _base_xform = transforms.ToTensor()
    def __init__(
            self,
            data_root: Union[Path, str],
            color: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
            seed: int=123,
            train: bool=True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.color = color.lower()
        # self._check_color_name() #TODO:
        assert self.color in ["gray", "red", "green", "blue"], "color must be one of gray, red, green, blue"

        # If transform is given, append it to the base transform
        self.transform = transforms.Compose([self._base_xform,
                                             Monochromizer(self.color)])
        if transform is not None:
            self.transform = transforms.Compose([self.transform, transform])

        self.target_transform = target_transform
        self.seed = seed
        self.train = train
        self.mode = 'train' if self.train else 'test'

        # Load four split MNIST subsets as a dictionary of
        # k=digit_id (int) and v=List[PIL.Image]
        bn = self.make_basename()
        fn = self.data_root / bn
        if not fn.exists() and download:
            print(f"{fn} doesn't exist)"
                  f"\nStart processing to split the MNIST dataset into 4 subsets...")
            mnist_data_root = data_root.parent
            self.save_split_mnist(mnist_data_root, data_root, seed=self.seed,
                                  use_train_dataset=train)

        self.data, self.targets = joblib.load(fn)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (pil_image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # img is a PIL Image of mode ('L'); unit8 in [0,...,255]; shape (28,28)

        if self.transform is not None:  # always perform the required basic transforms (ie. to Tensor and monochromizer)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def name(self) -> str:
        return self.make_basename()[:-4] # Drop suffix

    def make_basename(self):
        return self._fn_formatspec.format(mode=self.mode,
                                          color=self.color,
                                          seed=self.seed)

    @classmethod
    def save_split_mnist(cls,
                         mnist_data_root: Union[str,Path],
                         out_dir: Union[str,Path]=None,
                         seed: Optional[int]=None,
                         use_train_dataset:bool=True):
        """
        Split the original MNIST dataset into 4 non-overlapping subsets of (almost)
        equal size, using a random-split
        TODO:
        :param mnist_data_root: A directory that contains MNIST/raw and MNIST/processed.
            Eg. '/data/hayley-old/Tenanbaum2000/data/'
        :param out_dir: A directory to save the dictionries as pickle files.
            Eg. Path('/data/hayley-old/Tenanbaum2000/data/Mono-MNIST/')
        :param seed: Use an int to set a manual seed for a random generator that splits the
            origianl MNIST dataset into 4 subsets. Default: None
        :param use_train_dataset: True to use MNIST's training dataset, else use test dataset
        :return: None; Saves the 4 subset datasets as a dictionary of
            key=digit_id (int) and value=List[PIL.Image]
        """
        mode = 'train' if use_train_dataset else 'test'
        ds = datasets.MNIST(mnist_data_root, train=use_train_dataset, download=True)
        digits = defaultdict(list)
        for i in range(len(ds)):
            x, y = ds[i]  # PIL.Image of mode 'L' (ie. grayscale); y is int
            digits[y].append(x)
        print("Original MNIST: num images per digit")
        for k, v in digits.items():
            print(k, len(v))
        print("="*10)

        dict_gray = {}
        dict_r = {}
        dict_g = {}
        dict_b = {}
        for digit, imgs in digits.items():
            n = len(imgs)
            n_color = n // 4
            n_gray = n - 3 * n_color
            gray_imgs, r_imgs, g_imgs, b_imgs = random_split(imgs,
                                                           [n_gray, n_color, n_color, n_color],
                                                           generator=torch.Generator().manual_seed(seed))
            # gray/r/g/b: list of pil images -- each dataset has same digit id; different styles (either grayscale, red, green, blue)

            print("Digit id: ", digit)
            print("\t", [len(i) for i in [gray_imgs, r_imgs, g_imgs, b_imgs]])

            dict_gray[digit] = gray_imgs
            dict_r[digit] = r_imgs
            dict_g[digit] = g_imgs
            dict_b[digit] = b_imgs

        X_gray, Y_gray = cls.dict_imgs2tuple_xy(dict_gray)
        X_r, Y_r = cls.dict_imgs2tuple_xy(dict_r)
        X_g, Y_g = cls.dict_imgs2tuple_xy(dict_g)
        X_b, Y_b = cls.dict_imgs2tuple_xy(dict_b)

        def XY(color):
            return {
                "gray": (X_gray, Y_gray),
                "red": (X_r, Y_r),
                "green": (X_g, Y_g),
                "blue": (X_b, Y_b)
            }[color]

        for color in ['gray', 'red', 'green', 'blue']:
            bn = cls._fn_formatspec.format(mode=mode,
                                        color=color,
                                        seed=seed)
            out_fn = Path(out_dir) / bn
            joblib.dump(XY(color), out_fn)
            print("Saved: ", out_fn)

    @staticmethod
    def dict_imgs2tuple_xy(dict_imgs: Dict[int, List],
                           verbose=False) -> Tuple[Iterable, Iterable]:
        pil_imgs = []
        ys = []
        for k, v in dict_imgs.items():
            n_imgs = len(v)
            pil_imgs.extend(v)
            ys.extend([k] * n_imgs)

            if verbose:
                print("\n", k)
                print(len(pil_imgs), len(ys))

        return (pil_imgs, ys)