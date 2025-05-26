# registry_pattern.py

from typing import Dict, Type


class TypeRegistry:
    def __init__(self):
        self._types: Dict[str, Type] = {}

    def register(self, name: str, cls: Type) -> None:
        self._types[name] = cls

    def get(self, name: str) -> Type:
        return self._types[name]

    def all(self) -> Dict[str, Type]:
        return self._types
