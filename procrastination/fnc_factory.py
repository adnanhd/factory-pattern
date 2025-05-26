import inspect

from types import new_class
from typing_extensions import ParamSpec
from typing import Callable, Dict, Any, TypeVar, Generic, Tuple, Type
from functools import partial, partialmethod

from pydantic import BaseModel, validate_call, ValidationError, create_model

from registry import FunctionalRegistry, ObjectConfigMap

R = TypeVar("R")
P = ParamSpec("P")


def create_model_factory(t: Type) -> BaseModel:
    return create_model(
        # t.__module__ + "." +
        t.__qualname__ + "Builder",
        **{
            name: (
                Any if param.annotation == inspect._empty else param.annotation,
                ... if param.default == inspect._empty else param.default,
            )
            for name, param in inspect.signature(t).parameters.items()
        },
    )


class FunctionFactory(FunctionalRegistry[P, R], Generic[P, R]):
    """
    A factory for managing and executing registered functions dynamically.
    """

    @classmethod
    def create_builder(cls, func_name: str, /, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Execute a registered function by its key with the provided arguments.

        Args:
            func_key: The key representing the registered function.
            **kwargs: Arguments for the function.

        Returns:
            The result of the function execution.

        Raises:
            KeyError: If the function is not registered.
            ValidationError: If the arguments are invalid for the function.
        """
        func: Callable[P, R] = cls.get_registry_item(func_name)

        # Validate the function arguments
        func_name = (
            func.__qualname__
            if not isinstance(func, (partial, partialmethod))
            else func.func.__qualname__
        )
        builder = create_model(
            func_name + "Builder",
            **{
                name: (
                    Any if param.annotation == inspect._empty else param.annotation,
                    ... if param.default == inspect._empty else param.default,
                )
                for name, param in inspect.signature(func).parameters.items()
            },
        )
        return builder

    @classmethod
    def validate_function(cls, func_key: str, /, *args, **kwargs: Any) -> None:
        """
        Validate the inputs for a registered function.

        Args:
            func_key: The key representing the registered function.
            **kwargs: Arguments to validate.

        Raises:
            ValidationError: If the inputs are invalid.
        """
        # func = cls.get_registry_item(func_key)

        # # Validate function arguments using Pydantic
        # func = validate_call(func)
        # func(**kwargs)

    @classmethod
    def serialize_function(cls, func_key: str) -> Any:
        """
        Serialize the result of a registered function.

        Args:
            func_key: The key representing the registered function.

        Returns:
            Serialized result of the function.

        Raises:
            ValueError: If the result is not managed by the factory.
        """
        if func_key not in cls._config_map.keys():
            raise ValueError(
                f"Function '{func_key}' result is not managed by the factory."
            )
        return cls._config_map.get_registry_item(func_key)


# Example Usage
class FooFunctionRegistry(FunctionFactory[[int, int], int]): ...


# Registering functions
@FooFunctionRegistry.register_function
def add(a: int, b: int) -> int:
    return a + b


@FooFunctionRegistry.register_function
def concat_strings(s1: int, s2: int) -> int:
    return s1 + s2


# Example Execution
# Execute functions
# result1 = FooFunctionRegistry.execute_function("add", 5, 3)
# result2 = FooFunctionRegistry.execute_function("concat_strings", 10, s2=15)

print(result1)  # Output: 8
print(result2)  # Output: "Hello World"

# # Validate function inputs
# try:
#     FooFunctionRegistry.validate_function("add", a=5, b="not_an_int")
# except ValidationError as e:
#     print(f"Validation Error: {e}")

# Serialize function results
# serialized_result1 = FooFunctionRegistry.serialize_function("add")
# print(serialized_result1)  # Output: 8
