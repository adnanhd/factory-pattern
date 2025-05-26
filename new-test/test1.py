# main.py

from pydantic import validate_call

from factory import Factory
from registry_pattern import TypeRegistry

# --- Define your types ---


class Engine:
    def __init__(self, horsepower: int):
        self.horsepower = horsepower

    def __repr__(self):
        return f"Engine({self.horsepower} HP)"


class Robot:
    def __init__(self, name: str, engine: Engine):
        self.name = name
        self.engine = engine

    def __repr__(self):
        return f"Robot(name={self.name}, engine={self.engine})"

    @classmethod
    def from_config(cls, cfg):
        # Important: unwrap the nested factory object
        return cls(name=cfg.name, engine=cfg.engine.obj)


# --- Set up the registry and factory ---

registry = TypeRegistry()
registry.register("engine", Engine)
registry.register("robot", Robot)

Factory.registry = registry  # Inject the registry into Factory

# --- Example usage with validate_call ---


@validate_call
def build(x: Factory):
    print("Built:", x.obj)
    return x


# --- Run test ---

if __name__ == "__main__":
    f = build(
        {
            "type": "robot",
            "name": "Atlas",
            "engine": {"type": "engine", "horsepower": 300},
        }
    )
