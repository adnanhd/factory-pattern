from typing_compat import Annotated, List
from pydantic import BeforeValidator, WrapValidator, WrapSerializer, InstanceOf
import torch


def validate_positive_int_list(value: List[int]) -> torch.Size:
    """Ensure the input is a list of positive integers and convert it to torch.Size."""
    if not all(isinstance(dim, int) and dim > 0 for dim in value):
        raise ValueError("All dimensions must be positive integers.")
    return torch.Size(value)


def validate_torch_size(value: torch.Size, info) -> torch.Size:
    """Ensure the value is a valid torch.Size object."""
    if not isinstance(value, torch.Size):
        raise TypeError(f"Expected torch.Size, got {type(value).__name__}.")
    return value


def serialize_torch_size(value: torch.Size, handler) -> List[int]:
    """Serialize a torch.Size object into a list of integers."""
    return list(handler(value))


def validate_str_dtype(value: str, info) -> torch.dtype:
    """Ensure the value is a valid torch.dtype."""
    if not isinstance(value, str):
        raise TypeError(f"Expected torch.dtype, got {type(value).__name__}.")
    dtype_mapping = {
        "torch.float32": torch.float32,
        "torch.float": torch.float32,
        "torch.float64": torch.float64,
        "torch.double": torch.float64,
        "torch.float16": torch.float16,
        "torch.half": torch.float16,
        "torch.int32": torch.int32,
        "torch.int": torch.int32,
        "torch.int64": torch.int64,
        "torch.long": torch.int64,
        "torch.int16": torch.int16,
        "torch.short": torch.int16,
        "torch.int8": torch.int8,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
    }
    if value in dtype_mapping:
        return dtype_mapping[value]
    raise ValueError(f"{type(value).__name__} is not in the dtype_mapping")


def validate_torch_dtype(value: torch.dtype, info) -> torch.dtype:
    """Ensure the value is a valid torch.dtype."""
    if not isinstance(value, torch.dtype):
        raise TypeError(f"Expected torch.dtype, got {type(value).__name__}.")
    return value


def serialize_torch_dtype(value: torch.dtype, handler) -> str:
    """Serialize a torch.Size object into a list of integers."""
    return handler(str(value))


TorchDType = Annotated[
    InstanceOf[torch.dtype],
    WrapValidator(validate_torch_dtype),
    BeforeValidator(validate_str_dtype),
    WrapSerializer(serialize_torch_dtype),
]

TorchSize = Annotated[
    InstanceOf[torch.Size],
    WrapValidator(validate_torch_size),
    BeforeValidator(validate_positive_int_list),
    WrapSerializer(serialize_torch_size),
]
