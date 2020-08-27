from .. import CarbonModelBase

import numpy as np
import pandas as pd
import os
#from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class LinearRegression(CarbonModelBase):
    """
    Linear regression model. By default does not use the 'weight' feature in training or uses it in spesific ways as follows, weight =
        - 'd' (default), drops the weight feature (column) and does not use it in training,
        - 'm', drops the samples (rows) from training where the weight feature is missing
        - 'a', uses also the samples (rows) where weigth feature is missing.
    """
    def __init__(self):
        self.fit_intercept = True
        self.__set_filename()
        self.model = None

    def __set_filename(self):
        self.filename = f"linear_reg"

    def __preprocess(self, X):
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
        print('Split to testing data')

        # Initialize and train linear model
        model = linear_model.LinearRegression()
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
        return self.model.predict(X)