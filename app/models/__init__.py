
from abc import ABC, abstractmethod

class CarbonModelBase(ABC):    
    @abstractmethod
    def train(save_to=None):
        pass


# Import your models here
from .dummy import DummyModel


# Add your model to AVAILABLE_MODELS as name: Model entry. 
# Name is used in the command line to select the model.
AVAILABLE_MODELS = {'dummy': DummyModel}