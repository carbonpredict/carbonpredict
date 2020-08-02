import os, sys
import argparse
import pandas as pd
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


def prepare_data(model_name, dataset_id=None):
    X = get_data(dataset_id)
    X = X[~X["co2_total"].isna()]

    y = X["co2_total"].copy()
    X = X.drop("co2_total", axis=1)

    return X, y


def do_train(model_name, dataset_id=None, base_dir=None):
    X, y = prepare_data(model_name, dataset_id)
    AVAILABLE_MODELS[model_name]().train(X, y, base_dir)


def do_eval(model_name, dataset_id=None):
    X, y = prepare_data(model_name, dataset_id)
    AVAILABLE_MODELS[model_name]().eval(X, y)

def do_prediction(model_name, csv_file, base_dir):
    model = AVAILABLE_MODELS[model_name]()
    model.load(base_dir)

    X = pd.read_csv(csv_file)

    return model.predict(X)

def load_model(model_name):
    base_dir = os.environ.get('MNT_DIR', './')
    model = AVAILABLE_MODELS[model_name]()
    model.load(base_dir)
    return model

def do_prediction_with_params(model, params):
    X = pd.DataFrame.from_records([params])
    print('Dataframe before reindexing')
    print(X)
    X = X.reindex(sorted(X.columns), axis=1)
    print('Dataframe for prediction')
    print(X)
    print('Data types')
    print(X.dtypes)
    return model.predict(X)

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
    
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('model', type=str, help='Select model')
    predict_parser.add_argument('csv_file', type=str, help='CSV file')
    
    server_parser = subparsers.add_parser('run-server')
    
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()

    base_dir = os.environ.get('MNT_DIR', './')

    if args.subcommand == "train":
        if args.model in AVAILABLE_MODELS:
            if args.eval:
                do_eval(args.model)
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
