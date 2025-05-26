from pydantic import create_model
import inspect
from typing import Any, Type
import torch


def create_model_factory(t: Type):
    return create_model(
        # t.__module__ + "." +
        t.__qualname__ + "Config",
        **{
            name: (
                Any if param.annotation == inspect._empty else param.annotation,
                ... if param.default == inspect._empty else param.default,
            )
            for name, param in inspect.signature(t).parameters.items()
        }
    )


foo = create_model_factory(torch.nn.Linear)
