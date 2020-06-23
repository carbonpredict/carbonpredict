import os, sys
import argparse
import pandas as pd
from models import AVAILABLE_MODELS


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


def do_train(model_name, dataset_id=None, save_to=None):
    X, y = prepare_data(model_name, dataset_id)
    AVAILABLE_MODELS[model_name]().train(X, y, save_to)


def do_eval(model_name, dataset_id=None):
    X, y = prepare_data(model_name, dataset_id)
    AVAILABLE_MODELS[model_name]().eval(X, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Carbon Models')
    subparsers = parser.add_subparsers(title='subcommands', 
                                    description='valid subcommands', 
                                    help='Run subcommand --help for details',
                                    dest="subcommand")

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('model', type=str, help='Select model')
    train_parser.add_argument('--eval', action="store_true", help='Evaluate model')
    
    server_parser = subparsers.add_parser('run_server')
    # TODO: add server functionality

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()

    if args.subcommand == "train":
        if args.model in AVAILABLE_MODELS:
            if args.eval:
                do_eval(args.model)
            else:
                do_train(args.model)
        else:
            print(f"Error: model {args.model} is not available")



