from pydantic import validate_call

from factory import Factory
from mock_models import MockModel
from registry_pattern import TypeRegistry
from resnet_config import ResNet18Config

# Setup
registry = TypeRegistry()
registry.register("resnet18", MockModel)

Factory.registry = registry
Factory.register_validator("resnet18", ResNet18Config)


# Function using config via Factory
@validate_call
def build_model(cfg: Factory):
    model = cfg.instantiate()
    print("✨ Model:", model)

    if cfg.validate(model):
        print("✅ Validation passed")
    else:
        print("❌ Validation failed")

    print("📦 Serialized config:", cfg.serialize(model))


if __name__ == "__main__":
    build_model({"type": "resnet18", "pretrained": True, "layers": 18})
