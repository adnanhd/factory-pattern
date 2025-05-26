from pydantic.functional_validators import InstanceOf
import torch
from typing_extensions import Annotated
from pydantic.main import BaseModel
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, VisionDataset
from typing import Protocol, TypeVar, Tuple, Dict, Hashable, runtime_checkable
from datasets.metadata import BatchMetadata

T_co = TypeVar("T_co", covariant=True)
T_tuple = Tuple[T_co, ...]
T_dict = Dict[str, T_co]
T_batch = TypeVar("T_batch", T_tuple, T_dict)


@runtime_checkable
class _TorchDataset(Protocol[T_co]):
    def __getitem__(self, index: int) -> T_co: ...

    def __len__(self) -> int: ...


ff = BatchMetadata(
    shape=[32, 32, 3],
    dim_order=["b", "h", "w", "c"],
    data_repr="categorical",
    data_task="classification",
    data_device="cpu",
)


class MyDataset(Dataset):
    def __init__(self, dataset: _TorchDataset):
        self.dataset = dataset

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)


my_dataset = MyDataset(CIFAR10(root="/tmp", download=True, train=True))


from numpydantic import NDArray, Shape
from numpy import int8, int64


class Test(BaseModel):
    f: NDArray[Shape["* x, * y"], int64]
