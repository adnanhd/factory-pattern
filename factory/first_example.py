import inspect
from typing import Any, Type, Hashable, Generic, TypeVar
from pydantic import BaseModel, create_model

# Assume your existing TypeRegistry is imported from your project.
from registry import TypeRegistry

# Type variables for keys and registered types.
T = TypeVar("T")


def create_model_factory(t: Type) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model class for validating constructor arguments
    for the given type 't'.

    The model's fields are generated from the signature of 't'. For each parameter,
    if no annotation is provided, the field type defaults to Any, and if no default
    value is provided, the field is required (using ellipsis).

    Parameters:
        t (Type): The class type for which to create a builder model.

    Returns:
        Type[BaseModel]: A Pydantic model class representing the constructor schema.
    """
    return create_model(
        t.__qualname__ + "Builder",
        **{
            name: (
                Any if param.annotation == inspect._empty else param.annotation,
                ... if param.default == inspect._empty else param.default,
            )
            for name, param in inspect.signature(t).parameters.items()
        }
    )


class TypeFactory(TypeRegistry[T], Generic[T]):
    """
    Extension of TypeRegistry that implements a factory pattern.

    In addition to the standard registry functionality, TypeFactory provides the
    method 'instantiate_artifact', which validates keyword arguments against a
    dynamically generated Pydantic model (built from the registered type's signature)
    and instantiates the artifact if validation passes.
    """

    @classmethod
    def instantiate_artifact(cls, key: K, **kwargs: Any) -> T:
        """
        Instantiate an artifact (object) from the registered type using validated parameters.

        This method first retrieves the registered type corresponding to 'key' using
        get_registry_item (inherited from TypeRegistry). It then creates a builder model
        (via create_model_factory) based on the constructor signature of the type, and
        validates the provided kwargs. If validation passes, an instance of the type is
        created and returned.

        Parameters:
            key (K): The registry key corresponding to the desired type.
            **kwargs (Any): Keyword arguments to be validated and passed to the constructor.

        Returns:
            T: An instance of the registered type.

        Raises:
            pydantic.ValidationError: If the provided kwargs do not match the expected schema.
        """
        # Retrieve the registered type using the existing registry lookup.
        t = cls.get_registry_item(key)
        # Dynamically generate a Pydantic model for validating constructor parameters.
        builder_model = create_model_factory(t)
        # Validate the provided kwargs against the generated model.
        validated = builder_model.model_validate(kwargs)
        # Instantiate and return the object using the validated parameters.
        return t(**validated.model_dump())


# -----------------------------------------------------------------------------
# Example usage (for illustration)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example class to register.
    class MyClass:
        def __init__(self, a: int, b: int = 0):
            self.a = a
            self.b = b

        def compute(self) -> int:
            return self.a + self.b

    # Create a concrete TypeFactory for MyClass.
    class MyClassFactory(TypeFactory[MyClass]):
        _repository = {}

    # Register MyClass under a key.
    MyClassFactory.add_registry_item("MyClass", MyClass)

    # Now instantiate MyClass using the factory method.
    instance = MyClassFactory.instantiate_artifact("MyClass", a=10, b=5)
    print("Instance compute result:", instance.compute())
