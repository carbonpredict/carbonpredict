

from . import CarbonModelBase

class DummyModel(CarbonModelBase):
    def train(self, X, y, save_to=None):
        print("Training the model")
    
    
    def eval(self, X, y):
        print("Evaluating the model")

    