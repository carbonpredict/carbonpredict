import argparse

from models import DummyModel, AVAILABLE_MODELS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Carbon Models')
    parser.add_argument('model', type=str, help='Select model')
    
    args = parser.parse_args()

    if args.model in AVAILABLE_MODELS:
        AVAILABLE_MODELS[args.model]().train()    
    else:
        print(f"Error: model {args.model} is not available")






