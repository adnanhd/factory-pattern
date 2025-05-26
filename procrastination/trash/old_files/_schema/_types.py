""""""

from typing import Annotated, Literal
from annotated_types import Ge
import inspect
import torch

TorchDeviceIndexT = Annotated[int, Ge(0)] | None

TorchDeviceTypeT = Literal["cpu", "cuda", "xla", "mps", "ipu", "xpu"]
