"""
Deneme file for testing the pydantic_pytorch package.
"""

from typing import Protocol
from pydantic import BaseModel
import torch

from registry import make_class_registry, PydanticBuilderValidatorMacro


# bug with arguments with default values
class TorchModule(Protocol):
    """A torch module."""

    in_features: int
    out_features: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


TorchModuleRegistry = make_class_registry("TorchModule", TorchModule)


TorchModuleRegistry.register(torch.nn.Module)
TorchModuleRegistry.register_class(torch.nn.Linear)


class TrainModel(BaseModel):
    model: PydanticBuilderValidatorMacro[torch.nn.Module, TorchModuleRegistry]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)


model_from_instance = TrainModel.model_validate({"model": torch.nn.Linear(1, 1)})
model_from_config = TrainModel.model_validate_json(
    b'{"model": {"type": "Linear", "in_features": 1, "out_features": 1}}'
)
