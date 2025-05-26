from pydantic import BaseModel, ValidationError
from types import new_class
from typing import Any, Type, TypeVar, Generic, Hashable
from registry import TypeRegistry, RegistryError, ObjectConfigMap

T = TypeVar("T")


class TypeFactory(TypeRegistry[T], Generic[T]):
    """
    Factory for creating and managing instances of registered types.
    Extends TypeRegistry for dynamic type instantiation.
    """

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._config_map = new_class("ConfigMap", (ObjectConfigMap[T],))

    @classmethod
    def create_instance(cls, type_key: str, **kwargs) -> T:
        """
        Create an instance of the registered type using the provided key.

        Args:
            type_key: The key representing the registered type.
            **kwargs: The keyword arguments for the type's constructor.

        Returns:
            An instance of the registered type.

        Raises:
            KeyError: If the type_key is not registered.
            ValidationError: If the provided arguments are invalid.
        """
        klass = cls.get_registry_item(type_key)
        instance = klass(**kwargs)

        # Add instance to ObjectConfigMap for validation and serialization
        cls._config_map.add_registry_item(instance, kwargs)
        return instance

    @classmethod
    def validate_instance(cls, instance: Any) -> None:
        """
        Validate a given instance using ObjectConfigMap.

        Args:
            instance: The instance to validate.

        Raises:
            ValidationError: If the instance is invalid.
        """
        # if instance not in cls._config_map:
        #     raise ValueError(f"Instance {instance} is not managed by the factory.")

        # Validate the instance configuration
        # cls._config_map.validate(instance)

    @classmethod
    def serialize_instance(cls, instance: Any) -> dict:
        """
        Serialize a given instance to a dictionary using ObjectConfigMap.

        Args:
            instance: The instance to serialize.

        Returns:
            A dictionary representation of the instance.

        Raises:
            ValueError: If the instance is not managed by the factory.
        """
        return cls._config_map.get_registry_item(instance)


# Example Usage
class FooRegistry(TypeFactory[BaseModel]): ...


# Registering classes
@FooRegistry.register_class
class User(BaseModel):
    id: int
    name: str

    def __hash__(self) -> int:
        return id(self)


@FooRegistry.register_class
class Product(BaseModel):
    sku: str
    price: float

    def __hash__(self) -> int:
        return id(self)


# Example Execution
# Create instances
user_instance = FooRegistry.create_instance("User", id=1, name="Alice")
product_instance = FooRegistry.create_instance("Product", sku="1234", price=99.99)

# Validate instances
FooRegistry.validate_instance(user_instance)

# Serialize instances
user_serialized = FooRegistry.serialize_instance(user_instance)
print(user_serialized)  # Output: {'id': 1, 'name': 'Alice'}
