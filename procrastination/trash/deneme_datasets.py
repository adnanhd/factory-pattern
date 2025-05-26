from pydantic.functional_validators import AfterValidator
from datasets.example import cifar10_test_builder, cifar10_train_builder
from datasets.metadata import BatchMetadata
from torch.utils.data import Dataset
from pydantic import validate_call
from pydantic.dataclasses import dataclass
from pydantic import TypeAdapter, InstanceOf
from pydantic_core import SchemaValidator, core_schema
import torch

from typing import Annotated, Tuple


dataset = cifar10_train_builder()


def _batch_validator():
    print("foo")
    base_schema = core_schema.is_instance_schema(torch.Tensor)

    def _validate(v, hdlr):
        print("validated")
        return hdlr(v)

    return core_schema.no_info_wrap_validator_function(_validate, base_schema)


def _after_validate(v):
    print("validated")
    return v


a, b = dataset[12]

BatchType = Annotated[
    Tuple[Annotated[torch.Tensor, InstanceOf, AfterValidator(_after_validate)], int],
    SchemaValidator(_batch_validator()),
    # BatchMetadata(
    #     shape=[32, 32, 3],
    #     dim_order=["b", "h", "w", "c"],
    #     data_repr="categorical",
    #     data_task="classification",
    #     data_device="cpu",
    # ),
]


ta = TypeAdapter(BatchType)
