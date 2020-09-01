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
    
    training_samples: the number of samples (from beginning of dataframe) to use for training (default 100 000). Set using set_training_samples(samples).
    """
    def __init__(self):
        self.n_neighbors = 9
        self.training_samples = 100000
        self.__set_filename()
        self.model = None
    
    def __set_filename(self):
        self.filename = f"k_nearest_neighbors-k_{self.n_neighbors}_training_samples_{self.training_samples}.model"

    def __preprocess(self, X):
        # Drop empty features (dataset v. 1.0.0): unspsc_code, label 
        X = X.drop(["label", "unspsc_code"], axis=1)

        columns_to_include = ["category-1", "category-2", "category-3", "fabric_type", "size"]
        X = pd.DataFrame(columns=columns_to_include, data=X[columns_to_include].values)
        
        # Use unordered caterogies for several columns. List category values to support use cases when some
        # values are absent from a batch of source data.
        #brand_types = CategoricalDtype(categories=["b0", "b1", "b10", "b100", "b101", "b102", "b103", "b104", "b105", "b106", "b107", "b108", "b109", "b11", "b110", "b111", "b112", "b113", "b114", "b115", "b116", "b117", "b118", "b119", "b12", "b120", "b121", "b122", "b123", "b124", "b125", "b126", "b127", "b128", "b129", "b13", "b130", "b131", "b132", "b133", "b134", "b135", "b136", "b137", "b138", "b139", "b14", "b140", "b141", "b142", "b143", "b144", "b145", "b146", "b147", "b148", "b149", "b15", "b16", "b17", "b18", "b19", "b2", "b20", "b21", "b22", "b23", "b24", "b25", "b26", "b27", "b28", "b29", "b3", "b30", "b31", "b32", "b33", "b34", "b35", "b36", "b37", "b38", "b39", "b4", "b40", "b41", "b42", "b43", "b44", "b45", "b46", "b47", "b48", "b49", "b5", "b50", "b51", "b52", "b53", "b54", "b55", "b56", "b57", "b58", "b59", "b6", "b60", "b61", "b62", "b63", "b64", "b65", "b66", "b67", "b68", "b69", "b7", "b70", "b71", "b72", "b73", "b74", "b75", "b76", "b77", "b78", "b79", "b8", "b80", "b81", "b82", "b83", "b84", "b85", "b86", "b87", "b88", "b89", "b9", "b90", "b91", "b92", "b93", "b94", "b95", "b96", "b97", "b98", "b99"], ordered=False)
        #X["brand"] = X["brand"].astype(brand_types)
        cat1_types =  CategoricalDtype(categories=["baby", "clothing", "home", "kidswear", "menswear", "womenswear"], ordered=False)
        X["category-1"] = X["category-1"].astype(cat1_types)
        cat2_types = CategoricalDtype(categories=["home", "footwear", "nightwear", "thermals", "outerwear", "accessory", "uniform", "suit", "swimwear", "headgear", "sportswear", "costume", "clothing", "undergarments", "baby", "dress", "beachwear", "men-undergarments", "hosiery", "women-beachwear", "women-undergarments", "women-sportswear"], ordered=False)
        X["category-2"] = X["category-2"].astype(cat2_types)
        cat3_types = CategoricalDtype(categories=["backpack", "bikin", "body", "boxer-brief", "bra", "brief", "briefs", "cap", "coats", "costume", "curtain", "dress", "evening-dress", "fancy-dress", "flat-cap", "gloves", "hat", "hoodie", "jacket", "jean-shorts", "jeans", "jersey", "knit-cap", "knitwear", "long-sleeved-top", "mat", "overalls", "panties", "pants", "pillow", "pyjama", "scarf", "sheets", "shorts", "skirts", "snow-suit", "socks", "sport-bra", "stockings", "swimsuit", "T-shirt", "tie", "tights", "top", "towel", "trousers", "underpants", "wedding-dress"], ordered=False)
        X["category-3"] = X["category-3"].astype(cat3_types)
        #colour_types = CategoricalDtype(categories=["Ivory", "amber", "aquamarine", "black", "blue", "blue gray", "bondi blue", "brown", "colourful", "dark green", "dark grey", "gold", "golden", "gray", "green", "grey", "indigo", "light brown", "light grey", "lime", "maroon", "metal", "mosaic", "mustard", "natural", "navy", "neon", "orange", "peach", "pink", "purple", "red", "silver", "teal", "turquoise", "unbleached", "unknown", "violet", "wheat", "white", "yellow"], ordered=False)
        #X["colour"] = X["colour"].astype(colour_types)
        fabric_type_types = CategoricalDtype(categories=["K", "W"], ordered=False)
        X["fabric_type"] = X["fabric_type"].astype(fabric_type_types)
        #gender_types = CategoricalDtype(categories=["B", "G", "K", "M", "U", "Y", "W"], ordered=False)
        #X["gender"] = X["gender"].astype(gender_types)
        #made_in_types = CategoricalDtype(categories=["AU", "BD", "BE", "BG", "BR", "CN", "CO", "CY", "DE", "DK", "EG", "ES", "FI", "FR", "GB", "GE", "GR", "HK", "IE", "IN", "IT", "JP", "KR", "LT", "LV", "ML", "MX", "PK", "RO", "SE", "TH", "TR", "TW", "US", "VE", "VN"], ordered=False)
        #X["made_in"] = X["made_in"].astype(made_in_types)
        #season_types = CategoricalDtype(categories=["AYR", "MID", "SUM", "WIN"], ordered=False)
        #X["season"] = X["season"].astype(season_types)

        # Use ordered categories for size
        size_type = CategoricalDtype(categories=["XS", "S", "M", "L", "XL", "XXL"], ordered=True)
        X["size"] = X["size"].astype(size_type)

        # Convert the categoricals into a one-hot vector of binary variables
        X = pd.get_dummies(X)
        print('Data preprocessed')
        return X

    def __save_model(self, base_dir):
        print(f"Saving K-nearest neighbors model to disk at {base_dir}/{self.filename}")
        joblib.dump(self.model, f"{base_dir}/{self.filename}")

    def __train(self, X_train, y_train):
        # Only use the set number of samples for training
        if (len(X_train.index) > self.training_samples):
            X_train = X_train[:self.training_samples]
            y_train = y_train[:self.training_samples]

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.__preprocess(X_train)

        model = neighbors.KNeighborsRegressor(self.n_neighbors, weights='uniform')
        print('Model initialized. Starting to train model')
        model.fit(X_train, y_train)

        return model

    def __evaluate(self, X_test, y_test, model=None):
        if not model:
            model = self.model

        X_test = self.__preprocess(X_test)
        preds = model.predict(X_test)

        s_rmse = np.sqrt(mean_squared_error(y_test, preds))
        s_r2 = r2_score(y_test, preds)

        return s_rmse, s_r2, preds

    def set_n_neighbors(self, k):
        """
        Set the k to use in k-nearest neigbors (default k = 9)

        @param k (int): Number of neighbors to use in the k-nearest neigbors algorithm
        """
        self.n_neighbors = k
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

    def train(self, X_train, X_test, y_train, y_test, base_dir=None):
        print(f"Training K-nearest neighbors model with k = {self.n_neighbors} using {self.training_samples} samples")
        model = self.__train(X_train, y_train)
        self.model = model

        self.__save_model(base_dir)

    def eval(self, X_test, y_test):
        print(f"Evaluating K-nearest neighbors model with k = {self.n_neighbors} using training with {self.training_samples} samples")
        
        s_rmse, s_r2, y_pred = self.__evaluate(X_test, y_test)
        print(f"K-nearest neighbors trained with stats RMSE = {s_rmse}, R2 = {s_r2}")

        return s_r2, s_rmse, y_pred

    def predict(self, X):
        X = self.__preprocess(X)
        return self.model.predict(X)