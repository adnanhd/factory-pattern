from pydantic import GetCoreSchemaHandler, SerializerFunctionWrapHandler
from typing import Any
from typing import Any, Annotated, Literal
from functools import partial
from pydantic_core import core_schema, CoreSchema
from pydantic.json_schema import JsonSchemaValue
import torch
import inspect
import re
from .torch_dtype import TORCH_DTYPE_TYPE
from .torch_device import TORCH_DEVICE_TYPE


def serialize_py_torch_tensor(
    x: torch.Tensor, nxt: SerializerFunctionWrapHandler
) -> torch.Tensor:
    return nxt(x)


def torch_tensor_schema(
    dtype: TORCH_DTYPE_TYPE | None = None,
    device: TORCH_DEVICE_TYPE | None = None,
    strict: bool = True,
) -> CoreSchema:

    schema = core_schema.is_instance_schema(torch.Tensor)
    constraints = []

    def validate_torch_device_(x: torch.device) -> torch.device:
        str_x = _dtype_to_str(x)
        if str_x != dtype:
            raise ValueError(f"Invalid dtype: {str_x!r} " f"must be {dtype!r}")
        return x

    if dtype is not None:
        constraints.append(
            core_schema.no_info_after_validator_function(
                function=validate_torch_device_, schema=schema
            )
        )

    deserializer = core_schema.no_info_before_validator_function(
        function=partial(validate_py_torch_tensor, strict=strict), schema=schema
    )

    python_schema = core_schema.chain_schema([deserializer, schema, *constraints])
    python_schema["serialization"] = core_schema.wrap_serializer_function_ser_schema(
        serialize_py_torch_tensor, schema=core_schema.str_schema(), info_arg=False
    )

    deserializer = core_schema.no_info_before_validator_function(
        function=validate_js_torch_tensor, schema=schema
    )
    json_schema = core_schema.chain_schema([deserializer, schema, *constraints])
    json_schema["serialization"] = core_schema.wrap_serializer_function_ser_schema(
        serialize_js_torch_tensor, schema=core_schema.str_schema(), info_arg=False
    )

    return core_schema.json_or_python_schema(
        json_schema=json_schema, python_schema=python_schema
    )
