from .. import CarbonModelBase

import pandas as pd
import numpy as np

from pandas.api.types import CategoricalDtype
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class KNearestNeighbors(CarbonModelBase):
    """
    K-nearest neighbors clustering regression model. Currently uses only five columns from the input data:
    "category-1", "category-2", "category-3", "fabric_type", "size".
    
    The class has two important variables: 
    
    n_neighbors: the number of nearest neigbors to use (default 9) for the algorithm. Set using set_n_neighbors(k).
    
    training_samples: the number of samples (from beginning of dataframe) to use for training (default 50 000). Set using set_training_samples(samples).
    """
    def __init__(self):
        self.n_neighbors = 9
        self.training_samples = 1000
        self.__set_filename()
        self.model = None
    
    def __set_filename(self):
        self.filename = f"k-nearestneighbor_k_{self.n_neighbors}_training_samples_{self.training_samples}.model"

    def __preprocess(self, X):
        # Drop empty features (dataset v. 1.0.0): unspsc_code, label 
        X = X.drop(["label", "unspsc_code"], axis=1)

        # Use ordered categories for size
        size_type = CategoricalDtype(categories=["XS", "S", "M", "L", "XL", "XXL"], ordered=True)
        X["size"] = X["size"].astype(size_type)

        columns_to_include = ["category-1", "category-2", "category-3", "fabric_type", "size"]
        X = pd.DataFrame(columns=columns_to_include, data=X[columns_to_include].values)

        # Convert the categoricals into a one-hot vector of binary variables
        X = pd.get_dummies(X)

        return X

    def __save_model(self, base_dir):
        joblib.dump(self.model, f"{base_dir}/{self.filename}")

    def __train(self, X, y):
        # Only use the set number of samples for training
        if (len(DataFrame.index) > self.training_samples):
            X = X[:self.training_samples]
            y = y[:self.training_samples]
        
        X = self.__preprocess(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = neighbors.KNeighborsRegressor(self.n_neighbors, weights='uniform')
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        s_rmse = np.sqrt(mean_squared_error(y_test, preds))
        s_r2 = r2_score(y_test, preds)

        return model, s_r2

    def set_n_neighbors(self, k):
        """
        Set the k to use in k-nearest neigbors (default k = 9)

        @param k (int): Number of neighbors to use in the k-nearest neigbors algorithm
        """
        self.n_neighbors = n
        self.__set_filename()
    
    def set_training_samples(self, samples):
        """
        Set the number of samples to use in training the k-nearest neigbors model (default samples = 500 000).
        Note that a training is very slow with very large samples sizes.

        @param samples (int): Number of samples to use in training the k-nearest neigbors algorithm
        """
        self._training_samples = samples
        self.__set_filename()

    def load(self, base_dir):
        self.model = joblib.load(f"{base_dir}/{self.filename}")

    def train(self, X, y, base_dir=None):
        print(f"Training K-nearest neigbors model with k = {self.n_neighbors} using {self.training_samples} samples")
        model, _ = self.__train(X, y)
        self.__save_model()

    def eval(self, X, y):
        _, s_r2 = self.__train(X, y)
        return s_r2

    def predict(self, X):
        X = self.__preprocess(X)
        return self.model.predict(X)