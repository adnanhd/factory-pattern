class MockModel:
    def __init__(self, pretrained: bool, layers: int):
        self.pretrained = pretrained
        self.layers = layers
        self.parameters = [f"param{i}" for i in range(layers)]

    def __repr__(self):
        return f"MockModel(pretrained={self.pretrained}, layers={self.layers})"
