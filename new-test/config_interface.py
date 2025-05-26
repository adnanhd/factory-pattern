# config_interface.py

from abc import ABC
from typing import Any, Type


class ConfigInterfaceMixin(ABC):
    type: str
    __type_target__: Type = None  # will be injected dynamically

    def instantiate(self) -> Any:
        if self.__type_target__ is None:
            raise NotImplementedError("No target type attached.")
        return self.__type_target__(**self.model_dump())

    def validate(self, obj: Any) -> bool:
        return isinstance(obj, self.__type_target__)

    def serialize(self, obj: Any) -> dict:
        return self.model_dump()
