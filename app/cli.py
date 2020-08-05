import os, sys
import argparse
import pandas as pd
import numpy as np
from models import AVAILABLE_MODELS
from models import AVAILABLE_MODELS
from server import cpapi

def get_data(dataset_id=None):
    clone_dir = "/tmp/emission_data"
    os.system(f'git clone https://github.com/Compensate-Operations/emission-sample-data.git {clone_dir}/')
    os.system(f'for i in {clone_dir}/datasets/textile-v1.0.0/*.tgz; do tar -zxvf "$i" -C {clone_dir}/ ;done')
    os.system(f'rm {clone_dir}/._textile-v1.0.0-5.csv')
    content = sorted(filter(lambda x: x.endswith(".csv"), os.listdir(clone_dir)))
    return pd.concat((pd.read_csv(f'{clone_dir}/{f}') for f in content))

def get_data_from_dir(dataset_id=None, local_data_dir=None):
    print(f"Using source data from local dir {local_data_dir}")
    content = sorted(filter(lambda x: x.endswith(".csv"), os.listdir(local_data_dir)))
    return pd.concat((pd.read_csv(f'{local_data_dir}/{f}') for f in content))

def prepare_data(model_name, dataset_id=None, local_data=False,local_data_dir=None):
    X = None
    if (local_data):
        X = get_data_from_dir(dataset_id, local_data_dir)
    else:
        X = get_data(dataset_id)
    
    X = X[~X["co2_total"].isna()]

    y = X["co2_total"].copy()
    X = X.drop("co2_total", axis=1)

    return X, y

def do_train(model_name, dataset_id=None, base_dir=None, local_data=False, local_data_dir=None):
    X, y = prepare_data(model_name, dataset_id, local_data, local_data_dir)
    AVAILABLE_MODELS[model_name]().train(X, y, base_dir)

def do_eval(model_name, dataset_id=None, local_data=False, local_data_dir=None):
    X, y = prepare_data(model_name, dataset_id, local_data, local_data_dir)
    r2_score = AVAILABLE_MODELS[model_name]().eval(X, y)
    print(f'Model {model_name} evaluated, r2-score {r2_score}')

def do_prediction(model_name, csv_file, base_dir):
    model = AVAILABLE_MODELS[model_name]()
    model.load(base_dir)

    X = pd.read_csv(csv_file)

    return model.predict(X)

def load_model(model_name):
    base_dir = os.environ.get('MODEL_DIR', './')
    model = AVAILABLE_MODELS[model_name]()
    model.load(base_dir)
    return model

def do_prediction_with_params(model, params):
    X = pd.DataFrame.from_records([params])
    X = X.reindex(sorted(X.columns), axis=1)
    
    # Replace whitespace-only strings with NaN
    X = X.replace(r'^\s*$', np.nan, regex=True)
    
    # Some initial data type conversions: ideally the models should be robust enough to take care of this, but trying it here for now
    # Note: in the first dataset completely empty columns label and unspc_code are defined as bools here (lgbm is apparently using them and breaks if they are not int, float or bool). 
    # They should be treated some other way. The K-NN model drops them before further processing, so data type does not matter for it.
    X = X.astype({'ftp_acrylic': 'float64', 'ftp_cotton': 'float64', 'ftp_elastane': 'float64', 'ftp_linen': 'float64', 'ftp_other': 'float64', 'ftp_polyamide': 'float64', 'ftp_polyester': 'float64', 'ftp_polypropylene': 'float64', 'ftp_silk': 'float64', 'ftp_viscose': 'float64', 'ftp_wool': 'float64', 'label': 'bool', 'unspsc_code': 'bool'})

    prediction = model.predict(X)
    # The models return a list of predictions as float64. For the server, we are only predicting one sample and need to return a string (not a list), so return the first and only member of the list as a string.
    prediction = str(prediction[0])
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Carbon Models')
    subparsers = parser.add_subparsers(title='subcommands', 
                                    description='valid subcommands', 
                                    help='Run subcommand --help for details',
                                    dest="subcommand")

    models_parser = subparsers.add_parser('models')
    
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('model', type=str, help='Select model')
    train_parser.add_argument('--eval', action="store_true", help='Evaluate model')
    train_parser.add_argument('--local_data', action="store_true", help='Use csv source data in directory /tmp/emission_data')
    
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('model', type=str, help='Select model')
    predict_parser.add_argument('csv_file', type=str, help='CSV file')
    
    server_parser = subparsers.add_parser('run-server')
    
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()

    base_dir = os.environ.get('MODEL_DIR', './')
    local_data_dir = os.environ.get('LOCAL_DATA_DIR', './emission_data/')

    if args.subcommand == "train":
        if args.model in AVAILABLE_MODELS:
            if args.eval:
                if (args.local_data):
                    do_eval(args.model, local_data=True, local_data_dir=local_data_dir)
                else:
                    do_eval(args.model)
            else:
                if (args.local_data):
                    do_train(args.model, base_dir=base_dir, local_data=True, local_data_dir=local_data_dir)
                else:
                    do_train(args.model, base_dir=base_dir)
        else:
            print(f"Error: model {args.model} is not available")
    elif args.subcommand == "predict":
        if args.model in AVAILABLE_MODELS:
            predictions = do_prediction(args.model, args.csv_file, base_dir)
            print(predictions)
        else:
            print(f"Error: model {args.model} is not available")
    elif args.subcommand == "models":
        print("Available models:", list(AVAILABLE_MODELS.keys()))
        sys.exit(0)
    elif args.subcommand == "run-server":
        print("Starting web server...")
        cpapi.run()
