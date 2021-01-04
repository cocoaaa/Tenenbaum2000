import inspect
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from skimage.color import rgb2gray
from skimage.transform import resize

from pprint import pprint
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable
import warnings
from ipdb import set_trace

def now2str():
    now = datetime.now()
    now_str = now.strftime("%Y%m%d-%H%M%S")
    return now_str

def print_mro(x, print_fn:Callable=print):
    """
    Get the MRO of either a class x or an instance x
    """
    if inspect.isclass(x):
        [print_fn(kls) for kls in x.mro()[::-1]]
    else:
        [print_fn(kls) for kls in x.__class__.mro()[::-1]]

def info(arr, header=None):
    if header is None:
        header = "="*30
    print(header)
    print("shape: ", arr.shape)
    print("dtype: ", arr.dtype)
    print("min, max: ", min(np.ravel(arr)), max(np.ravel(arr)))

def mkdir(p: Path, parents=True):
    if not p.exists():
        p.mkdir(parents=parents)
        print("Created: ", p)


def get_next_version(save_dir:Union[Path,str], name:str):
    """Get the version index for a file to save named in pattern of
    f'{save_dir}/{name}/version_{current_max+1}'

    Get the next version index for a directory called
    save_dir/name/version_[next_version]
    """
    root_dir = Path(save_dir)/name

    if not root_dir.exists():
        warnings.warn("Returning 0 -- Missing logger folder: %s", root_dir)
        return 0

    existing_versions = []
    for p in root_dir.iterdir():
        bn = p.stem
        if p.is_dir() and bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace('/', '')
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1


def get_next_version_path(save_dir: Union[Path, str], name: str):
    """Get the version index for a file to save named in pattern of
    f'{save_dir}/{name}/version_{current_max+1}'

    Get the next version index for a directory called
    save_dir/name/version_[next_version]
    """
    root_dir = Path(save_dir) / name

    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)
        print("Created: ", root_dir)

    existing_versions = []
    for p in root_dir.iterdir():
        bn = p.stem
        if p.is_dir() and bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace('/', '')
            existing_versions.append(int(dir_ver))

    if len(existing_versions) == 0:
        next_version = 0
    else:
        next_version = max(existing_versions) + 1

    return root_dir / f"version_{next_version}"
