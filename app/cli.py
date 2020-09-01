import os, sys
import argparse
import pandas as pd
import numpy as np
from models import AVAILABLE_MODELS
from server import cpapi
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_data(repo_url, repo_data_directory, data_format='tgz', dataset_id=None):
    clone_dir = '/tmp/emission_data'
    print(f'Cloning repo {repo_url} to directory {clone_dir}...')
    os.system(f'git clone {repo_url} {clone_dir}/')
    
    if (data_format=='tgz'):
        print(f'Unzipping with tar...')
        os.system(f'for i in {clone_dir}/{repo_data_directory}/*.tgz; do tar -zxvf "$i" -C {clone_dir}/ ;done')
    elif (data_format=='csv'):
        os.system(f'for i in {clone_dir}/{repo_data_directory}/*.csv; do cp "$i" {clone_dir}/ ;done')
    else:
        raise ValueError('Source data format not recognized. Only tgz and csv supported.')
    
    # Remove known garbage file in textile source data v. 1.0.0
    garbage_file = f'{clone_dir}/._textile-v1.0.0-5.csv'
    if (os.path.isfile(garbage_file)):
        print(f'Removing garbage file {garbage_file}')
        os.system(f'rm {garbage_file}')
    
    content = sorted(filter(lambda x: x.endswith('.csv'), os.listdir(clone_dir)))
    return pd.concat((pd.read_csv(f'{clone_dir}/{f}') for f in tqdm(content, desc="Reading csv")))

def get_data_from_dir(local_data_dir=None, dataset_id=None):
    print(f'Using source data from local dir {local_data_dir}')
    content = sorted(filter(lambda x: x.endswith('.csv'), os.listdir(local_data_dir)))
    return pd.concat((pd.read_csv(f'{local_data_dir}/{f}') for f in tqdm(content, desc="Reading csv")))

def prepare_data(local_data=False, local_data_dir=None, repo_url=None, repo_data_directory=None, data_format='tgz', dataset_id=None, random_state=42):
    X = None

    print("Loading csv files, this may take a while...")

    if (local_data):
        X = get_data_from_dir(local_data_dir,dataset_id)
    else:
        X = get_data(repo_url, repo_data_directory, data_format, dataset_id)
    
    X = X[~X['co2_total'].isna()]

    y = X['co2_total'].copy()
    X = X.drop('co2_total', axis=1)

    print('Split to training and testing data')
    return train_test_split(X, y, test_size=0.2, random_state=random_state)

def do_train(model_name, base_dir=None, local_data=False, local_data_dir=None,
    repo_url='https://github.com/Compensate-Operations/emission-sample-data.git', 
    repo_data_directory='datasets/textile-v1.0.0', 
    data_format='tgz',
    dataset_id=None,
    random_state=42,
    save_test=False):

    if (base_dir == None):
        base_dir = os.environ.get('MODEL_DIR', './')

    X_train, X_test, y_train, y_test = prepare_data(local_data, local_data_dir, repo_url, repo_data_directory, data_format, dataset_id, random_state)
    print('Data preparation complete. Starting training of model.')
    model = AVAILABLE_MODELS[model_name]()
    model.train(X_train, X_test, y_train, y_test, base_dir)
    

def do_eval(model_name, local_data=False, local_data_dir=None,
    repo_url='https://github.com/Compensate-Operations/emission-sample-data.git', 
    repo_data_directory='datasets/textile-v1.0.0', 
    data_format='tgz', 
    dataset_id=None,
    random_state=42,
    save_test=False):

    X_train, X_test, y_train, y_test = prepare_data(local_data, local_data_dir, repo_url, repo_data_directory, data_format, dataset_id, random_state)
    preds = {}
    print('Data preparation complete. Starting training and evaluation of model.')
    model = AVAILABLE_MODELS[model_name]()
    model.train(X_train, X_test, y_train, y_test, base_dir)
    r2_score, rmse_score, y_pred = model.eval(X_test, y_test)
    preds[model_name]=y_pred
    if save_test:
        pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True), pd.DataFrame(preds).reset_index(drop=True)], axis=1, ignore_index=True).to_csv(f'/results/test_pred_{model_name}.csv')
    print(f'Model {model_name} evaluated, r2-score {r2_score}')

    return r2_score

def evaluate_available_models(local_data=False, local_data_dir=None,
    repo_url='https://github.com/Compensate-Operations/emission-sample-data.git', 
    repo_data_directory='datasets/textile-v1.0.0', 
    data_format='tgz', 
    dataset_id=None,
    random_state=42,
    save_test=False):
    
    X_train, X_test, y_train, y_test = prepare_data(local_data, local_data_dir, repo_url, repo_data_directory, data_format, dataset_id, random_state)
    pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1, ignore_index=True).to_csv(f'/results/testset.csv')
    preds = {}
    print('Data preparation complete. Starting training and evaluation of model.')
    for model_name in AVAILABLE_MODELS:
        if model_name == 'k_nearest_neighbors': continue
        print(f'****{model_name}****')
        model = AVAILABLE_MODELS[model_name]()
        model.train(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), base_dir)
        r2_score, rmse_score, y_pred = model.eval(X_test.copy(), y_test.copy())
        preds[model_name] = y_pred
        print(f'Model {model_name} evaluated, r2-score {r2_score}')
        pd.DataFrame(preds).to_csv(f'/results/test_pred_{model_name}.csv')
    if save_test:
            pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True), pd.DataFrame(preds).reset_index(drop=True)], axis=1, ignore_index=True).to_csv(f'/results/test_pred_all.csv')

def format_predictions(predictions, intervals=False):
    """Ensure that predictions are non-negative and format them to a python list (or list of lists for the intervals option)"""
    predictions = pd.DataFrame(predictions).applymap(lambda x: max(x, 0))
    
    if intervals and predictions.shape[1] != 3:
        print(f'Warning: model returned an incorrect number of values per sample (should be 3, was {predictions.shape[1]}). Returning tuples containing all values per sample.')
        predictions = predictions.values.tolist()
    elif intervals:
        print('Returning triplets (mean, 5-percentile, 95-percentile)')
        predictions = predictions.values.tolist()
    elif predictions.shape[1] > 1:
        print('Warning: model returned more than one value per sample, using only the first value for each sample')
        predictions = predictions.iloc[:, 0].tolist()
    else: 
        predictions = predictions.iloc[:, 0].tolist()
    return predictions

def do_prediction(model_name, csv_file, base_dir, intervals=False):
    model = AVAILABLE_MODELS[model_name]()
    model.load(base_dir)

    X = pd.read_csv(csv_file)

    predictions = model.predict(X)
    predictions = format_predictions(predictions, intervals)
    return predictions

def evaluate_trained_models(csv_file, base_dir):
    trained_models = get_trained_models()
    test_data = pd.read_csv(csv_file)
    y_true = test_data['co2_total'].copy()

    dtypes = np.dtype([
          ('Model name', str),
          ('RMSE', float),
          ('R2', float),
          ])
    data = np.empty(0, dtype=dtypes)
    results = pd.DataFrame(data)

    for idx, model_name in enumerate(trained_models):
        preds = do_prediction(model_name, csv_file, base_dir)
        y_pred = pd.DataFrame(preds)
        results.at[idx, 'Model name'] = model_name
        results.at[idx,'RMSE'] = mean_squared_error(y_true, y_pred, squared=False)
        results.at[idx,'R2'] = r2_score(y_true, y_pred)

    print (results)

def get_models():
    return list(AVAILABLE_MODELS.keys())

def get_trained_models():
    """
    The model files must begin with the exact name of the model (defined in AVAILABLE_MODELS) followed by a short hyphen -
    and must use the file type .model (e.g. k_nearest_neighbors-k_9_training_samples_100000.model).
    """
    base_dir = os.environ.get('MODEL_DIR', './')
    model_files = sorted(filter(lambda x: x.endswith('.model'), os.listdir(base_dir)))
    model_names = sorted(set(map(lambda n: n.split('-')[0], model_files)))
    return model_names

def load_model(model_name):
    base_dir = os.environ.get('MODEL_DIR', './')
    model = AVAILABLE_MODELS[model_name]()
    model.load(base_dir)
    return model

def convert_params_to_df(params):
    X = pd.DataFrame.from_records([params])
    X = X.reindex(sorted(X.columns), axis=1)
    
    # Replace whitespace-only strings with NaN
    X = X.replace(r'^\s*$', np.nan, regex=True)
    
    # Some initial data type conversions: ideally the models should be robust enough to take care of this, but trying it here for now
    # Note: in the first dataset completely empty columns label and unspc_code are defined as bools here (lgbm is apparently using them and breaks if they are not int, float or bool). 
    # They should be treated some other way. The K-NN model drops them before further processing, so data type does not matter for it.
    X = X.astype({'ftp_acrylic': 'float64', 'ftp_cotton': 'float64', 'ftp_elastane': 'float64', 'ftp_linen': 'float64', 'ftp_other': 'float64', 'ftp_polyamide': 'float64', 'ftp_polyester': 'float64', 'ftp_polypropylene': 'float64', 'ftp_silk': 'float64', 'ftp_viscose': 'float64', 'ftp_wool': 'float64', 'label': 'bool', 'unspsc_code': 'bool'})
    return X

def do_prediction_with_params(model, params, intervals=False):
    X = convert_params_to_df(params)
    predictions = model.predict(X)
    
    predictions = format_predictions(predictions, intervals)
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Carbon Models')
    subparsers = parser.add_subparsers(title='subcommands', 
                                    description='valid subcommands', 
                                    help='Run subcommand --help for details',
                                    dest='subcommand')

    models_parser = subparsers.add_parser('models')

    models_parser = subparsers.add_parser('trained_models')
    
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('model', type=str, help='Select model')
    train_parser.add_argument('--eval', action='store_true', help='Evaluate model')
    train_parser.add_argument('--local_data', action='store_true', help='Use csv source data in directory /tmp/emission_data')
    train_parser.add_argument('--random_state', type=str, help='Random state for train-test split')
    train_parser.add_argument('--save_test', action='store_true', help='Store test set')
    
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('model', type=str, help='Select model')
    predict_parser.add_argument('csv_file', type=str, help='CSV file with data to predict for')
    predict_parser.add_argument('--intervals', action='store_true', help='Show 0.05 and 0.95 prediction intervals')
    
    evalmodels_parser = subparsers.add_parser('evaluate_trained_models')
    evalmodels_parser.add_argument('csv_file', type=str, help='CSV file with test data')

    evaltrainmodels_parser = subparsers.add_parser('retrain_evaluate_models')
    evaltrainmodels_parser.add_argument('--save_test', action='store_true', help='Store test sets')
    evaltrainmodels_parser.add_argument('--random_state', type=str, help='Random state for train-test split')

    server_parser = subparsers.add_parser('run-server')
    
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()

    base_dir = os.environ.get('MODEL_DIR', './')
    local_data_dir = os.environ.get('LOCAL_DATA_DIR', './emission_data/')

    if args.subcommand == 'train':
        if args.model in AVAILABLE_MODELS:
            if args.eval:
                if (args.local_data):
                    _ = do_eval(args.model, local_data=True, local_data_dir=local_data_dir, random_state=args.random_state, save_test=args.save_test)
                else:
                    _ = do_eval(args.model, random_state=args.random_state, save_test=args.save_test)
            else:
                if (args.local_data):
                    do_train(args.model, base_dir=base_dir, local_data=True, local_data_dir=local_data_dir, random_state=args.random_state, save_test=args.save_test)
                else:
                    do_train(args.model, base_dir=base_dir, random_state=args.random_state, save_test=args.save_test)
        else:
            print(f'Error: model {args.model} is not available')
    elif args.subcommand == 'predict':
        trained_models = get_trained_models()
        if args.model in trained_models:
            pd.set_option('display.max_rows', None)
            predictions = do_prediction(args.model, args.csv_file, base_dir, args.intervals)
            print(predictions)
        elif args.model in AVAILABLE_MODELS:
            print(f'Error: model {args.model} is available but not trained. Please train the model first.')
        else:
            print(f'Error: model {args.model} is not available, check available models with command models.')
    elif args.subcommand == 'models':
        print('Available models:', get_models())
        sys.exit(0)
    elif args.subcommand == 'trained_models':
        print('Trained models:', get_trained_models())
        sys.exit(0)
    elif args.subcommand == 'evaluate_trained_models':
        evaluate_trained_models(args.csv_file, base_dir)
        sys.exit(0)
    elif args.subcommand == 'retrain_evaluate_models':
        evaluate_available_models(local_data=True, local_data_dir=local_data_dir, random_state=args.random_state, save_test=args.save_test)
    elif args.subcommand == 'run-server':
        print('Starting web server...')
        cpapi.run()
