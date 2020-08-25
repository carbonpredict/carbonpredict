from .. import CarbonModelBase
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from lightgbm import LGBMRegressor

class LGBMDefault(CarbonModelBase):

    def __init__(self):
        self.models = []
        self.params = {'bagging_fraction': 0.4,
                'bagging_freq': 10,
                'boosting_type': 'gbdt',
                'colsample_bytree': 0.4,
                'lambda_l1': 0.0,
                'lambda_l2': 0.0,
                'learning_rate': 0.1,
                'max_depth': 12,
                'metric': 'rmse',
                'n_jobs': -1,
                'num_leaves': 300,
                'objective': 'regression',
                'seed': 42,
                'verbose': -1}
        self.n_splits = 5


    def __preprocess(self, X):
        cat_cols = ["category-1", "category-2", "category-3", 
            "size", "made_in", "gender", "colour", 
            "brand", "fabric_type", "season"]

        X[cat_cols] = X[cat_cols].astype("category")
        X = X.drop("weight", axis=1)

        return X
    
    def __train(self, X, y):
        print("Preprocessing data...")
        X = self.__preprocess(X)
        
        print("Training model...")
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        preds = np.zeros(len(X))
        nrounds = 5000
        early_stopping_rounds = 200

        models = []

        for fold, (trn_idx, val_idx) in enumerate(tqdm(kf.split(X, y), total = self.n_splits, desc="Folds")):
            X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
            X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

            trn_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_valid, label=y_valid)

            lgb_clf = lgb.train(self.params,
                            trn_data,
                            nrounds,
                            valid_sets = [trn_data, val_data],
                            early_stopping_rounds = early_stopping_rounds,
                            verbose_eval = False)

            preds[val_idx] = lgb_clf.predict(X_valid)

            models.append(lgb_clf)

        s_rmse = np.sqrt(mean_squared_error(y, preds))
        s_r2 = r2_score(y, preds)
        
        return models, s_r2


    def __get_filename(self, fold):
        return f"lgbm_default-{fold}.model"


    def load(self, base_dir):
        self.models = []

        for idx in tqdm(range(self.n_splits)):
            self.models.append(lgb.Booster(model_file=f"{base_dir}/{self.__get_filename(idx)}"))


    def train(self, X, y, base_dir=None):
        models, _ = self.__train(X, y)

        print(f"Saving model to {base_dir}/")

        for idx, model in enumerate(tqdm(models, total = self.n_splits, desc="Save")):
            model.save_model(f"{base_dir}/{self.__get_filename(idx)}", 
                            num_iteration=model.best_iteration)


    def eval(self, X, y):
        _, s_r2 = self.__train(X, y)

        return s_r2


    def predict(self, X):
        X = self.__preprocess(X)
        X = X.drop("co2_total", axis=1)
        return list(np.mean([model.predict(X) for model in self.models], axis=0))




class LGBMQuantileRegression(LGBMDefault):
    def __init__(self):
        super().__init__()

        self.qreg_low = None
        self.qreg_mid = None
        self.qreg_high = None

    
    def __train_qreg(self, X, y):
        self.params["objective"] = "quantile"

        quantile_alphas = [0.05, 0.5, 0.95]

        lgb_quantile_alphas = []
        for quantile_alpha in tqdm(quantile_alphas, desc="Training quantiles"):
            lgb = LGBMRegressor(alpha=quantile_alpha, **self.params)
            lgb.fit(X, y)
            lgb_quantile_alphas.append(lgb)

        return lgb_quantile_alphas


    def __get_qreg_filename(self, c):
        return f"lgbm_qreg-{c}.model"


    def train(self, X, y, base_dir=None):
        X = self._LGBMDefault__preprocess(X)
        
        qreg_low, qreg_mid, qreg_high = self.__train_qreg(X, y)

        qreg_low.booster_.save_model(f"{base_dir}/{self.__get_qreg_filename('low')}")
        qreg_mid.booster_.save_model(f"{base_dir}/{self.__get_qreg_filename('mid')}")
        qreg_high.booster_.save_model(f"{base_dir}/{self.__get_qreg_filename('high')}")


    def load(self, base_dir):
        self.qreg_low = lgb.Booster(model_file=f"{base_dir}/{self.__get_qreg_filename('low')}")
        self.qreg_mid = lgb.Booster(model_file=f"{base_dir}/{self.__get_qreg_filename('mid')}")
        self.qreg_high = lgb.Booster(model_file=f"{base_dir}/{self.__get_qreg_filename('high')}")


    def predict(self, X):
        X = self._LGBMDefault__preprocess(X)
        X = X.drop("co2_total", axis=1)

        low_pred = self.qreg_low.predict(X)
        mid_pred = self.qreg_mid.predict(X)
        high_pred = self.qreg_high.predict(X)

        return pd.DataFrame({'prediction': mid_pred, 'lower_ci': low_pred, 'higher_ci': high_pred})

