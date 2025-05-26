from torch_models import TorchSize, TorchDType
import torch
from pydantic import validate_call
from registry.api import functional_registry_factory
from registry import FunctionalRegistry
import typing
import inspect

from types import new_class
from typing_extensions import ParamSpec
from typing import Callable, Dict, Any, TypeVar, Generic, Tuple, Type
from functools import partial, partialmethod

from pydantic import BaseModel, validate_call, ValidationError, create_model

from registry import FunctionalRegistry, ObjectConfigMap

R = TypeVar("R")
P = ParamSpec("P")


def create_model_factory(t: Type) -> BaseModel:
    return create_model(
        # t.__module__ + "." +
        t.__qualname__ + "Builder",
        **{
            name: (
                Any if param.annotation == inspect._empty else param.annotation,
                ... if param.default == inspect._empty else param.default,
            )
            for name, param in inspect.signature(t).parameters.items()
        }
    )


class TorchCreator(FunctionalRegistry[[TorchSize, TorchDType], torch.Tensor]):
    """TorchFunctionalLossRegistry"""


@TorchCreator.register_function
@validate_call
def zeros(shape: TorchSize, dtype: TorchDType) -> torch.Tensor:
    """Create a tensor of zeros with the given shape and dtype."""
    return torch.zeros(size=shape, dtype=dtype)


@TorchCreator.register_function
@validate_call
def ones(shape: TorchSize, dtype: TorchDType) -> torch.Tensor:
    """Create a tensor of ones with the given shape and dtype."""
    return torch.ones(size=shape, dtype=dtype)


@validate_call
def uniform(
    shape: TorchSize, dtype: TorchDType, low: float = 0.0, high: float = 1.0
) -> torch.Tensor:
    """Create a tensor with uniform random values."""
    return torch.empty(size=shape, dtype=dtype).uniform_(low, high)


@validate_call
def normal(
    shape: TorchSize, dtype: TorchDType, mean: float = 0.0, std: float = 1.0
) -> torch.Tensor:
    """Create a tensor with normal random values."""
    return torch.empty(size=shape, dtype=dtype).normal_(mean, std)
