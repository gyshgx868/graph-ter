from abc import abstractmethod


class Transformer:
    def __init__(self, out_features):
        self.out_features = out_features

    @abstractmethod
    def get_config(self):
        pass
