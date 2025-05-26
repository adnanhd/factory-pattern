# factory.py

from inspect import signature
from typing import Any, Callable, Dict, Type, get_type_hints

from pydantic import BaseModel, GetCoreSchemaHandler, create_model
from pydantic_core import core_schema

from config_interface import ConfigInterfaceMixin
from registry_pattern import TypeRegistry


class Factory:
    registry: TypeRegistry = None
    validators: Dict[str, Type[BaseModel]] = {}
    constructors: Dict[str, Callable] = {}

    @classmethod
    def register_validator(cls, name: str, validator: Type[BaseModel]):
        cls.validators[name] = validator

    @classmethod
    def register_constructor(cls, name: str, constructor: Callable):
        cls.constructors[name] = constructor

    @classmethod
    def create_base_model(cls, typ: Type) -> Type[BaseModel]:
        sig = signature(typ.__init__)
        hints = get_type_hints(typ.__init__)
        fields = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            annotation = hints.get(name)
            if annotation is None:
                raise TypeError(f"Missing annotation for parameter '{name}' in {typ}")
            fields[name] = (annotation, ...)

        model_cls = create_model(
            f"{typ.__name__}Config",
            __base__=(BaseModel, ConfigInterfaceMixin),
            **fields,
        )
        model_cls.__type_target__ = typ
        return model_cls

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def factory(value: Any) -> BaseModel:
            if not isinstance(value, dict):
                raise TypeError("Expected dict input")
            type_name = value.get("type")
            if not type_name:
                raise ValueError("Missing 'type' key in config")
            typ = cls.registry.get(type_name)
            validator = cls.validators.setdefault(type_name, cls.create_base_model(typ))
            return validator.model_validate(value)

        return core_schema.no_info_after_validator_function(
            factory, core_schema.any_schema()
        )
