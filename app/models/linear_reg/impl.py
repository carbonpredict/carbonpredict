from .. import CarbonModelBase

import numpy as np
import pandas as pd
import os
#from sklearn.linear_model import LinearRegression
from pandas.api.types import CategoricalDtype
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class LinearRegression(CarbonModelBase):
    """
    Linear regression model. Weight feature is dropped from from the training. Otherwise all other features are used.
    """
    def __init__(self):
        self.fit_intercept = True
        self.__set_filename()
        self.model = None

    def __set_filename(self):
        self.filename = f"linear_reg.model"

    def __preprocess(self, X):
        # Drop empty features (dataset v. 1.0.0): unspsc_code, label 
        X = X.drop(["label", "unspsc_code"], axis=1)

        X = X.drop('weight', axis=1)

        # Use unordered caterogies for several columns. List category values to support use cases when some
        # values are absent from a batch of source data.
        brand_types = CategoricalDtype(categories=["b0", "b1", "b10", "b100", "b101", "b102", "b103", "b104", "b105", "b106", "b107", "b108", "b109", "b11", "b110", "b111", "b112", "b113", "b114", "b115", "b116", "b117", "b118", "b119", "b12", "b120", "b121", "b122", "b123", "b124", "b125", "b126", "b127", "b128", "b129", "b13", "b130", "b131", "b132", "b133", "b134", "b135", "b136", "b137", "b138", "b139", "b14", "b140", "b141", "b142", "b143", "b144", "b145", "b146", "b147", "b148", "b149", "b15", "b16", "b17", "b18", "b19", "b2", "b20", "b21", "b22", "b23", "b24", "b25", "b26", "b27", "b28", "b29", "b3", "b30", "b31", "b32", "b33", "b34", "b35", "b36", "b37", "b38", "b39", "b4", "b40", "b41", "b42", "b43", "b44", "b45", "b46", "b47", "b48", "b49", "b5", "b50", "b51", "b52", "b53", "b54", "b55", "b56", "b57", "b58", "b59", "b6", "b60", "b61", "b62", "b63", "b64", "b65", "b66", "b67", "b68", "b69", "b7", "b70", "b71", "b72", "b73", "b74", "b75", "b76", "b77", "b78", "b79", "b8", "b80", "b81", "b82", "b83", "b84", "b85", "b86", "b87", "b88", "b89", "b9", "b90", "b91", "b92", "b93", "b94", "b95", "b96", "b97", "b98", "b99"], ordered=False)
        X["brand"] = X["brand"].astype(brand_types)
        cat1_types =  CategoricalDtype(categories=["baby", "clothing", "home", "kidswear", "menswear", "womenswear"], ordered=False)
        X["category-1"] = X["category-1"].astype(cat1_types)
        cat2_types = CategoricalDtype(categories=["home", "footwear", "nightwear", "thermals", "outerwear", "accessory", "uniform", "suit", "swimwear", "headgear", "sportswear", "costume", "clothing", "undergarments", "baby", "dress", "beachwear", "men-undergarments", "hosiery", "women-beachwear", "women-undergarments", "women-sportswear"], ordered=False)
        X["category-2"] = X["category-2"].astype(cat2_types)
        cat3_types = CategoricalDtype(categories=["backpack", "bikin", "body", "boxer-brief", "bra", "brief", "briefs", "cap", "coats", "costume", "curtain", "dress", "evening-dress", "fancy-dress", "flat-cap", "gloves", "hat", "hoodie", "jacket", "jean-shorts", "jeans", "jersey", "knit-cap", "knitwear", "long-sleeved-top", "mat", "overalls", "panties", "pants", "pillow", "pyjama", "scarf", "sheets", "shorts", "skirts", "snow-suit", "socks", "sport-bra", "stockings", "swimsuit", "T-shirt", "tie", "tights", "top", "towel", "trousers", "underpants", "wedding-dress"], ordered=False)
        X["category-3"] = X["category-3"].astype(cat3_types)
        colour_types = CategoricalDtype(categories=["Ivory", "amber", "aquamarine", "black", "blue", "blue gray", "bondi blue", "brown", "colourful", "dark green", "dark grey", "gold", "golden", "gray", "green", "grey", "indigo", "light brown", "light grey", "lime", "maroon", "metal", "mosaic", "mustard", "natural", "navy", "neon", "orange", "peach", "pink", "purple", "red", "silver", "teal", "turquoise", "unbleached", "unknown", "violet", "wheat", "white", "yellow"], ordered=False)
        X["colour"] = X["colour"].astype(colour_types)
        fabric_type_types = CategoricalDtype(categories=["K", "W"], ordered=False)
        X["fabric_type"] = X["fabric_type"].astype(fabric_type_types)
        gender_types = CategoricalDtype(categories=["B", "G", "K", "M", "U", "Y", "W"], ordered=False)
        X["gender"] = X["gender"].astype(gender_types)
        made_in_types = CategoricalDtype(categories=["AU", "BD", "BE", "BG", "BR", "CN", "CO", "CY", "DE", "DK", "EG", "ES", "FI", "FR", "GB", "GE", "GR", "HK", "IE", "IN", "IT", "JP", "KR", "LT", "LV", "ML", "MX", "PK", "RO", "SE", "TH", "TR", "TW", "US", "VE", "VN"], ordered=False)
        X["made_in"] = X["made_in"].astype(made_in_types)
        season_types = CategoricalDtype(categories=["AYR", "MID", "SUM", "WIN"], ordered=False)
        X["season"] = X["season"].astype(season_types)

        # Use ordered categories for size
        size_type = CategoricalDtype(categories=["XS", "S", "M", "L", "XL", "XXL"], ordered=True)
        X["size"] = X["size"].astype(size_type)

        # Convert the categoricals into a one-hot vector of binary variables
        X = pd.get_dummies(X)
        #print(X)

        # Fill in 0 for NA in ftp_ columns
        X = X.fillna(0)
        #print(X)


        return X

    def __preprocess_v1(self, X):
        # Drop empty features (dataset v. 1.0.0): unspsc_code, label 
        print('Start preprocessing data')

        # Set missing fiber type percentages to zero
        values ={'ftp_acrylic': 0, 'ftp_cotton': 0, 'ftp_elastane': 0, 'ftp_linen': 0, 'ftp_other': 0, 'ftp_polyamide': 0, 'ftp_polyester': 0, 'ftp_polypropylene': 0, 'ftp_silk': 0, 'ftp_viscose': 0, 'ftp_wool': 0}
        X = X.fillna(value=values)
        print('Null fiber percentages changed to zero')
    
        # Fill categorical nan values for gender and season features with mode values. May need to be updated with new training data
        X['gender'] = X.fillna(X['gender'].value_counts().index[0])
        X['season'] = X.fillna(X['season'].value_counts().index[0])
        print('Categorial values with null replaced with mode values')
    
        # Convert the categoricals into a one-hot vector of dummy binary variables
        X = pd.get_dummies(X,columns=['category-1', 'category-2', 'category-3', 'brand', 'colour', 'fabric_type', 'gender', 'season','made_in','size'], prefix = ['category-1', 'category-2', 'category-3', 'brand', 'colour', 'fabric_type',  'gender', 'season','made_in','size'])
        print('Categorial values changed to dummy one-hot vectors')
    
        # If still some null values, change them to zero. At least the weight feature (column) has many null values. 
        X = X.fillna(0)
        print('Rest of the null values set to zero. Particularly the missing weight values')

        return X

    def __save_model(self, base_dir):
        print(f"Saving Linear Regression model to disk at {base_dir}/{self.filename}")
        joblib.dump(self.model, f"{base_dir}/{self.filename}")

    def __train(self, X, y):
        print(f"Training linear regression model")


        print('Preprocess data')
        X = self.__preprocess(X)
        print('Data preprocessed')
        X = X.to_numpy(dtype='float32')
        y = y.to_numpy(dtype='float32')
        print('Formatted to numpy')
    
        # Split training data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print('Split to training and testing data')

        # Initialize and train linear model
        model = linear_model.LinearRegression(fit_intercept=True)
        print('Model initialized')
        model.fit(X_train, y_train)
        print('Model trained')
    
        # Make predictions based on the model
        y_fit = model.predict(X_test)
        print('Make predictions')
    
        # Evaluate model
        s_rmse = mean_squared_error(y_test, y_fit, squared=False)
        s_r2 = r2_score(y_test, y_fit)
        print(f"Linear model trained with stats RMSE = {s_rmse}, R2 = {s_r2}")

        return model, s_r2


    def load(self, base_dir):
        self.model = joblib.load(f"{base_dir}/{self.filename}")

    def train(self, X, y, base_dir=None):
        print(f"Training Linear Regression model with tbd.")
        model, _ = self.__train(X, y)
        self.model = model
        self.__save_model(base_dir)

    def eval(self, X, y):
        print(f"Evaluating Linear Regression model with tbd.")
        _, s_r2 = self.__train(X, y)
        return s_r2

    def predict(self, X):
        X = self.__preprocess(X)
        X = X.drop('co2_total', axis=1)
        return self.model.predict(X)