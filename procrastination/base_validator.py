from typing import Any, Generic, TypeVar, get_origin, get_args
from pydantic import BaseModel, ValidationError, field_validator
from functools import lru_cache
import torch

# Generic type T for validators
T = TypeVar("T")


class BaseValidator(BaseModel, Generic[T]):
    """Generic validator with validate and serialize methods."""

    value: T

    def __hash__(self):
        return id(self)

    @lru_cache
    def create(self):
        return torch.Size(self.value)

    @classmethod
    def validate(cls, val: T) -> T:
        """Validate the value based on the generic type T."""
        import pdb

        pdb.set_trace()
        origin = get_origin(cls.__annotations__["value"])
        args = get_args(cls.__annotations__["value"])

        if origin is list and all(isinstance(v, args[0]) for v in val):
            # Example validation for list types
            return val
        elif origin is torch.dtype and isinstance(val, torch.dtype):
            # Example validation for torch.dtype
            return val
        elif origin is torch.device and isinstance(val, torch.device):
            # Example validation for torch.device
            return val
        else:
            raise ValueError(
                f"Unsupported or invalid type for {cls.__name__}: {type(val).__name__}"
            )

    def serialize(self) -> dict:
        """Serialize the value into a dictionary."""
        return {"type": self.__class__.__name__, "value": str(self.value)}
