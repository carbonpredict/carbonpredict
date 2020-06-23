
from abc import ABC, abstractmethod

class CarbonModelBase(ABC):    
    @abstractmethod
    def train(X, y, save_to=None):
        """
        Train and save the model to
        a given location. 

        @param X Training data
        @param y Training labels
        @param save_to Directory to save model
        """
        pass

    @abstractmethod
    def eval(X, y):
        """
        Evaluate the model and return S^2 score. 

        @param X Training data
        @param y Training labels
        @return S^2 score 
        """
        pass


# Import your models here
from .dummy import DummyModel
from .lgbm_default import LGBMDefault

# Add your model to AVAILABLE_MODELS as name: Model entry. 
# Name is used in the command line to select the model.
AVAILABLE_MODELS = {'dummy': DummyModel, 
                    'lgbm_default': LGBMDefault}