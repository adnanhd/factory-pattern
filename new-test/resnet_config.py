from pydantic import BaseModel

from config_interface import ConfigInterfaceMixin
from mock_models import MockModel


class ResNet18Config(BaseModel, ConfigInterfaceMixin):
    type: str = "resnet18"
    pretrained: bool = True
    layers: int = 18

    def instantiate(self):
        return MockModel(pretrained=self.pretrained, layers=self.layers)

    def validate(self, obj):
        return (
            isinstance(obj, MockModel)
            and obj.pretrained == self.pretrained
            and len(obj.parameters) == self.layers
        )
