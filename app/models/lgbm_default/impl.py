from .. import CarbonModelBase
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

class LGBMDefault(CarbonModelBase):

    def __init__(self):
        self.models = []
        self.params = {'bagging_fraction': 1.0,
                'bagging_freq': 1,
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
        return f"{self.__class__.__name__}_{fold}.model"


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

