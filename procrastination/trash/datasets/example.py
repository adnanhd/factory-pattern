from annotated_types import Predicate
from pydantic_core import SchemaValidator, core_schema
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    ToTensor,
    RandomHorizontalFlip,
    RandomRotation,
    ColorJitter,
    Normalize,
)
from .__init__ import Dataset, WrapDataset
from torchdatasets.datasets import TorchDataset
import torchdatasets.maps as maps
import torchdatasets.cachers as cachers
from torch import Tensor as TorchTensor
import pathlib
from .metadata import BatchMetadata
from pydantic import InstanceOf, validate_call
from typing import Annotated, Tuple


ROOT_PREFIX = pathlib.Path("/tmp/torch-data/")
CACHE_PREFIX = pathlib.Path("/tmp/torch-cache/")

ff = BatchMetadata(
    shape=[32, 32, 3],
    dim_order=["b", "h", "w", "c"],
    data_repr="categorical",
    data_task="classification",
    data_device="cpu",
)


def _batch_validator():
    base_schema = core_schema.is_instance_schema(TorchTensor)

    def _validate(v, hdlr):
        print("validated")
        return hdlr(v)

    return core_schema.no_info_wrap_validator_function(_validate, base_schema)


class WrapPydanticDataset:
    dataset: Annotated[TorchDataset, InstanceOf]

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)  # type: ignore


@validate_call
def cifar10_train_builder() -> Annotated[
    TorchDataset[
        Annotated[
            Tuple[TorchTensor, int],
            SchemaValidator(_batch_validator()),
            # BatchMetadata(
            #     shape=[32, 32, 3],
            #     dim_order=["b", "h", "w", "c"],
            #     data_repr="categorical",
            #     data_task="classification",
            #     data_device="cpu",
            # ),
        ]
    ],
    InstanceOf,
]:
    train_set = WrapDataset(
        CIFAR10(root=ROOT_PREFIX / "cifar-10", train=True, download=True)
    )
    transforms = Compose(
        [
            ToTensor(),
            RandomHorizontalFlip(),
            RandomRotation(degrees=10),
            ColorJitter(brightness=0.5),
            Normalize(
                mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
                std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628),
            ),
        ]
    )
    return train_set.map(maps.To(transforms, 0)).cache(
        cachers.Pickle(CACHE_PREFIX / "cifar-10-train")  # type: ignore
    )


def cifar10_test_builder():
    test_set = WrapDataset(
        CIFAR10(root=ROOT_PREFIX / "cifar-10", train=False, download=True)
    )
    transforms = Compose(
        [
            ToTensor(),
            Normalize(
                mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
                std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628),
            ),
        ]
    )
    return test_set.map(maps.To(transforms, 0)).cache(
        cachers.Pickle(CACHE_PREFIX / "cifar-10-test")  # type: ignore
    )
