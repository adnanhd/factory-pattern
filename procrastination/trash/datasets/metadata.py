from typing import List, Literal, Annotated, Sequence
from annotated_types import Gt, Len
from pydantic import BaseModel, ConfigDict, Field, field_validator, constr
from enum import Enum

from pydantic_core.core_schema import CoreSchema
from pydantic_core import core_schema
from torchdatasets.datasets import Dataset
from annotated_types import BaseMetadata


class DataType(str, Enum):
    CATEGORICAL = "categorical"
    LOGISTICAL = "logistical"


class BatchMetadata(BaseModel):
    type: Literal["batch-metadata"] = "batch-metadata"

    shape: Annotated[List[Annotated[int, Gt(0)]], Len(1, 3)]
    """Shape must be a list of 1 to 3 positive integers (excluding batch size)."""

    dim_order: Annotated[Sequence[Literal["h", "w", "c", "b"]], Len(2, 4)]
    """Dimension order must be a list containing 'h', 'w', 'c', 'b' with 'b' mandatory."""

    data_repr: Literal["categorical", "logistical"]

    data_task: Literal["segmentation", "classification", "regression", "generative"]

    data_device: Literal["cpu", "gpu"]

    @field_validator("shape")
    def validate_shape(cls, v):
        if not all(isinstance(i, int) and i > 0 for i in v):
            raise ValueError("Each shape value must be a positive integer.")
        return v

    @field_validator("dim_order")
    def validate_dim_order(cls, v, values):
        if "b" not in v:
            raise ValueError("'b' (batch) must be included in dim_order.")
        if len(set(v)) != len(v):
            raise ValueError("Dimension order must not contain duplicate values.")
        if not set(v).issubset({"h", "w", "c", "b"}):
            raise ValueError("Dimension order must contain only 'h', 'w', 'c', 'b'.")
        if len(v) != len(values.data.get("shape", [])) + 1:
            raise ValueError("Dimension order must match with shape")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "shape": [32, 32, 3],
                "dim_order": ["b", "h", "w", "c"],
                "data_type": "logistical",
            }
        }
    )

    def generate_schema(self) -> CoreSchema:
        def _validate(v: Dataset):
            print("validated")
            return v

        return core_schema.no_info_before_validator_function(
            _validate, core_schema.is_instance_schema(Dataset)
        )
