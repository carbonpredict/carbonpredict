from .. import CarbonModelBase

import pandas as pd
import numpy as np
import torch

from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import os

class OneLayerModelRobust(nn.Module):
    def __init__(self, n_input, n_hidden1, n_output, bs, p):
        super().__init__()
        self.bs = bs
        self.drop_layer = nn.Dropout(p=p)
        self.input_layer = nn.Linear(n_input, n_hidden1)
        self.hidden1 = nn.Linear(n_hidden1, n_output)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.drop_layer(x)
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden1(x)
        return x

class NeuralNetworkOneLayerFFRobust:
    """
    A feedforward neural network model with one hidden layer that uses feature dropout during training. 
    The number of neurons in the hidden layer can be given as a parameter to the constructor (default 1024).
    The model will train until 5 epochs have passed without the test RMSE improving. 
    """
    def __init__(self, hidden_neurons=1024, learning_rate = 0.01, batch_size=1000, droprate=0.2):
        self.hidden_neurons = hidden_neurons
        self.lr = learning_rate
        self.bs = batch_size
        self.droprate = droprate
        self.__set_filename()
        self.model = None
    
    def __set_filename(self):
        self.filename = f"neural_onelayer_robust-hidden_{self.hidden_neurons}.model"

    #def preprocess(self, X):
    def __preprocess(self, X):
        # Drop empty features (dataset v. 1.0.0): unspsc_code, label 
        X = X.drop(["label", "unspsc_code"], axis=1)

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

        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        #print(X_scaled)

        return X_scaled

    def __save_model(self, model, base_dir):
        torch.save(model, f"{base_dir}/{self.filename}")

    def __get_dataloader(self, X_train, X_test, y_train, y_test, bs=1000, test_size=0.2):
        X_train = self.__preprocess(X_train)
        X_test = self.__preprocess(X_test)
        X_train = X_train.to_numpy(dtype='float32')
        X_test = X_test.to_numpy(dtype='float32')
        y_train = y_train.to_numpy(dtype='float32')
        y_test = y_test.to_numpy(dtype='float32')
        
        y_train = y_train.reshape((-1,1))
        y_test = y_test.reshape((-1,1))

        train_dataloader = DataLoader(TensorDataset(
            torch.tensor(X_train, dtype=torch.float),
            torch.tensor(y_train, dtype=torch.float)),
            shuffle=True,
            batch_size=bs)

        test_dataloader = DataLoader(TensorDataset(
            torch.tensor(X_test, dtype=torch.float),
            torch.tensor(y_test, dtype=torch.float)),
            shuffle=True,
            batch_size=bs)      

        return train_dataloader, test_dataloader
    
    def __evaluate(self, dataloader, model, criterion, device):
        model.eval()

        rmse_scores = []
        r2_scores = []
        losses = []

        with torch.no_grad():
            for batch in dataloader:
                X, y = batch
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)

                loss = criterion(y_pred, y)
                losses.append(loss)

                with torch.no_grad():
                    y_pred = y_pred.cpu()
                    y = y.cpu()
                    s_rmse = mean_squared_error(y, y_pred, squared=False)
                    s_r2 = r2_score(y, y_pred)

                    rmse_scores.append(s_rmse)
                    r2_scores.append(s_r2)

        model.train()

        return torch.mean(torch.tensor(losses)), torch.mean(torch.tensor(rmse_scores)), torch.mean(torch.tensor(r2_scores))

    def __train(self, train_dataloader, test_dataloader, model, optimizer, criterion, device, base_dir, write_model_to_disk=False):
        model.train()

        best_test_rmse_score = None
        best_test_r2_score = None
                
        fmt = '{:<5} {:12} {:12} {:<9} {:<9} {:<9} {:<9}'
        print(fmt.format('Epoch', 'Train loss', 'Valid loss', 'Train RMSE', 'Train R2', 'Test RMSE', 'Test R2'))

        epoch = 0
        best_score_epoch = 0
        while (epoch - best_score_epoch < 4):
            epoch = epoch + 1
        
            for i, batch in enumerate(train_dataloader):
                X, y = batch

                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

            
            train_loss, train_rmse_score, train_r2_score = self.__evaluate(train_dataloader, model, criterion, device)
            test_loss, test_rmse_score, test_r2_score = self.__evaluate(test_dataloader, model, criterion, device)

            fmt = '{:<5} {:03.2f} {:03.2f} {:02.2f} {:02.2f} {:02.2f} {:02.2f}'
            print(fmt.format(epoch, train_loss, test_loss, train_rmse_score, train_r2_score, test_rmse_score, test_r2_score))
        
            if ((best_test_rmse_score == None) or (test_rmse_score < best_test_rmse_score)):
                best_test_rmse_score = test_rmse_score
                best_test_r2_score = test_r2_score
                best_score_epoch = epoch
                self.model = model
                if (write_model_to_disk):
                    self.__save_model(model, base_dir)
            
        print(f"Neural network one hidden layer robust model trained in {best_score_epoch} epochs with stats RMSE = {best_test_rmse_score}, R2 = {best_test_r2_score}")
        print(f"Saved neural network one hidden layer model to disk at {base_dir}/{self.filename}")

        return best_test_r2_score
    
    def __select_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using GPU!")
        else:
            device = torch.device('cpu')
            print("GPU not available, using CPU.")
        return device
    
    def load(self, base_dir):
        print(f"Loading neural network one hidden layer model from disk at {base_dir}/{self.filename}")
        model = torch.load(f"{base_dir}/{self.filename}")
        model.eval()
        self.model = model

    def train(self, X_train, X_test, y_train, y_test, base_dir=None):
        device = self.__select_device()
        print(f"Preparing batches of training data")
        train_dataloader, test_dataloader = self.__get_dataloader(X_train, X_test, y_train, y_test)
        
        model = OneLayerModelRobust(334, self.hidden_neurons, 1, self.bs, p=self.droprate).to(device)
        
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) # Stochastic gradient descent
        #optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        criterion = nn.MSELoss(reduction='mean')
                
        print(f"Starting training of neural network one hidden layer robust model with {self.hidden_neurons} hidden layer neurons and batch size {self.bs}")
        _ = self.__train(train_dataloader, test_dataloader, model, optimizer, criterion, device, base_dir, write_model_to_disk=True)
        print(f"Training complete")

    def eval(self, X_test, y_test):
        # Make predictions based on the model
        print(f"Evaluating neural network one hidden layer model with {self.hidden_neurons} hidden layer neurons and batch size {self.bs}")
        y_fit = self.predict(X_test)
        #print('Predictions stored')
    
        # Evaluate model
        s_rmse = mean_squared_error(y_test, y_fit, squared=False)
        s_r2 = r2_score(y_test, y_fit)
        print(f"NN trained with stats RMSE = {s_rmse}, R2 = {s_r2}")
        return s_r2, s_rmse, y_fit

    def predict(self, X):
        X = X.drop(["co2_total"], axis=1, errors='ignore')
        device = self.__select_device()
        X = self.__preprocess(X)
        X = X.to_numpy(dtype='float32')
        X = torch.tensor(X, dtype=torch.float)
        X = X.to(device)        
        y_pred = self.model(X)
        y_pred = y_pred.detach().cpu().numpy().flatten().tolist()
        
        return y_pred