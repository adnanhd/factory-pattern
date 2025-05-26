"""
import pydantic_pytorch
import pydantic
import torch
from typing_extensions import TypedDict
from pydantic_pytorch.constructions.construction import to_typed_dict
from pydantic_pytorch.annotations.multiarray.torch_device import TorchDevice
from typing import Annotated
from pydantic import WrapSerializer, BeforeValidator, AfterValidator, InstanceOf, Field


class Validate:
    @classmethod
    def __class_getitem__(cls, item):
        return Annotated[item, InstanceOf()]



# CUDA = InstanceOf[TorchDevice]({'type': 'cuda', 'index': 0})

class Test(pydantic.BaseModel):
    device: TorchDevice
    model_config = pydantic.ConfigDict(extra='allow')


test = Test.model_validate({
    'device': {
        'type': 'cuda',
        'index': None
    }
})


"""

from pydantic import BaseModel, ConfigDict, TypeAdapter, NonNegativeInt, ValidationError
from pydantic_core import core_schema, SchemaValidator
from os import PathLike
from typing import Annotated
from typing_extensions import TypedDict, Literal, Required, Union, Any
from re import Pattern
from pydantic_pytorch.annotations.multiarray.torch_device import TORCH_DEVICE_TYPE


def _dict_not_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


class TorchDeviceSchema(TypedDict, total=False):
    type: Required[Literal["torch-device"]]
    to_device: TORCH_DEVICE_TYPE
    to_index: NonNegativeInt
    strict: bool


def torch_device_schema(
    type: Literal["torch-device"],
    to_device: TORCH_DEVICE_TYPE,
    to_index: NonNegativeInt,
    strict: bool,
) -> TorchDeviceSchema:
    return _dict_not_none(
        type=type, to_device=to_device, to_index=to_index, strict=strict
    )


def torch_device_core_schema() -> core_schema.CoreSchema:
    return {
        "type": "model",
        "model": {
            "type": "computed-field",
            "fields": {
                "type": {"type": "literal", "value": "torch-device"},
                "to_device": {
                    "type": "any"
                },  # Assuming Pydantic can handle `torch.device` type correctly
                "to_index": {"type": "int", "ge": 0},  # NonNegativeInt is an int >= 0
                "strict": {"type": "bool"},
            },
        },
    }


class Foo(BaseModel):
    a: str

    model_config = ConfigDict(baz="qux")


from typing import final
from typing_extensions import Self
from pydantic_core.core_schema import CoreSchema, CoreConfig

# core_schema.CoreConfig.register('torch-device', torch_device_core_schema())
try:
    pass
    # f = SchemaValidator({'type': 'torch-device', 'to_device': 'cuda', 'to_index': 0, 'strict': True})
    # f.validate_python(12)
except ValidationError as e:
    error = e
