from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, ValidationError, create_model


class ObjectRegistry:
    """A registry to store and retrieve object types."""

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, key: str, type_cls: Type):
        """Register a class with a key."""
        if key in cls._registry:
            raise ValueError(f"Key '{key}' is already registered.")
        cls._registry[key] = type_cls

    @classmethod
    def get(cls, key: str) -> Optional[Type]:
        """Retrieve a class by its key."""
        return cls._registry.get(key)


class Factory:
    """Factory to create objects with validated configuration."""

    def __init__(self, registry: Type[ObjectRegistry]):
        self.registry = registry

    def create(self, config_data: Dict[str, Any]) -> Any:
        """Create an object based on validated configuration."""
        obj_type = config_data.get("type")
        params = config_data.get("params", {})

        # Retrieve the class from the registry
        cls = self.registry.get(obj_type)
        if not cls:
            raise ValueError(f"Type '{obj_type}' is not registered.")

        # Validate the configuration using the class's schema
        if hasattr(cls, "get_config_schema"):
            schema = cls.get_config_schema()
            try:
                validated_params = schema(**params)
            except ValidationError as e:
                raise ValueError(f"Invalid configuration for '{obj_type}': {e}")
        else:
            validated_params = params

        # Create and return the object
        return cls(**validated_params.dict())


class MyObject:
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"MyObject(name={self.name}, value={self.value})"

    @staticmethod
    def get_config_schema() -> Type[BaseModel]:
        return create_model("MyObjectConfig", name=(str, ...), value=(int, ...))


class AnotherObject:
    def __init__(self, id: int, description: str):
        self.id = id
        self.description = description

    def __repr__(self):
        return f"AnotherObject(id={self.id}, description={self.description})"

    @staticmethod
    def get_config_schema() -> Type[BaseModel]:
        return create_model(
            "AnotherObjectConfig", id=(int, ...), description=(str, ...)
        )


# Register classes
ObjectRegistry.register("MyObject", MyObject)
ObjectRegistry.register("AnotherObject", AnotherObject)

# Create the factory
factory = Factory(ObjectRegistry)

# Valid configuration
config_data = {"type": "MyObject", "params": {"name": "example", "value": 42}}

instance = factory.create(config_data)
print(instance)

# Invalid configuration
invalid_config = {
    "type": "AnotherObject",
    "params": {"id": "not an int", "description": "example"},
}

try:
    invalid_instance = factory.create(invalid_config)
except ValueError as e:
    print(e)
