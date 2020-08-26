#!flask/bin/python
from flask import Flask, jsonify, abort, make_response, request, url_for
from werkzeug.utils import secure_filename
from cli import do_eval, do_prediction_with_params, do_train, get_models, get_trained_models, load_model
from flasgger import Swagger, swag_from
import os
import time


"""
Flask server and REST based prediction API and admin API for the CCaaS service. 

Example usage. 
1. Start the server (from repository root directory `docker-compose run --service-ports carbon run-server`)
2. Call prediction API with CURL using wsocks.json file with test input (from repo root directory `curl -i -H 'Content-Type: application/json' -X POST --data 'testdata/@wsocks.json' http://localhost:5000/ccaas/api/v0.1/predict`)
"""

app = Flask(__name__)

app.config['SWAGGER'] = {
    'title': 'Carbon predict API',
    'url': '/ccaas/api/v0.1',
    'version': '0.1',
    'uiversion': 3
}

swagger = Swagger(app)

@app.route('/ccaas/api/v0.1/predict', methods=['POST'])
@swag_from('predict.yml')
def predict():    
    """
    Predict CO2e for a product with given features
    """
    if (not request.json):
        abort(400)
     
    product = {
    'brand': request.json['brand'],
    'category-1': request.json['category-1'],
    'category-2': request.json['category-2'],
    'category-3': request.json['category-3'],
    'colour': request.json['colour'],
    'fabric_type': request.json['fabric_type'],
    'ftp_acrylic': request.json['ftp_acrylic'],
    'ftp_cotton': request.json['ftp_cotton'],
    'ftp_elastane': request.json['ftp_elastane'],
    'ftp_linen': request.json['ftp_linen'],
    'ftp_other': request.json['ftp_other'],
    'ftp_polyamide': request.json['ftp_polyamide'],
    'ftp_polyester': request.json['ftp_polyester'],
    'ftp_polypropylene': request.json['ftp_polypropylene'],
    'ftp_silk': request.json['ftp_silk'],
    'ftp_viscose': request.json['ftp_viscose'],
    'ftp_wool': request.json['ftp_wool'],
    'gender': request.json['gender'],
    'label': request.json['label'],
    'made_in': request.json['made_in'],
    'season': request.json['season'],
    'size': request.json['size'],
    'unspsc_code': request.json['unspsc_code'],
    'weight': request.json['weight'],
    'ML-model': request.json['ML-model']
    }

    product['co2_total'] = None
    ml_model = product.pop('ML-model', None)
    if (ml_model == None or ml_model == ''):
        print('Loading default model: LGBM')
        ml_model = 'lgbm_default'
    else:
        print(f'Loading model: {ml_model}')
    model = load_model(ml_model)
    print('Model loaded')
    
    pred_with_intervals = do_prediction_with_params(model, product, intervals=True)
    
    prediction = pred_with_intervals[0][0]
    percentile_5 = pred_with_intervals[0][1] if len(pred_with_intervals[0]) == 3 and pred_with_intervals[0][1] is not None else None
    percentile_95 = pred_with_intervals[0][2] if len(pred_with_intervals[0]) == 3 and pred_with_intervals[0][2] is not None else None
    result = {
        "prediction": prediction,
        "5-percentile": percentile_5,
        "95-percentile": percentile_95
    }
    
    print('CO2e prediction complete, returning result')
    print(result)
    
    resp = jsonify(result)
    resp.status_code = 201
    return resp

@app.route('/ccaas/api/v0.1/train', methods=['POST'])
@swag_from('train.yml')
def train_model():
    """
    Train a model with given source data
    """
    if (not request.json 
        or not 'data-format' in request.json
        or not 'ML-model' in request.json
        or not 'source-data-directory' in request.json
        or not 'source-data-repo' in request.json):
        abort(400)

    data_format = request.json['data-format']
    ML_model = request.json['ML-model']
    source_data_directory = request.json['source-data-directory']
    source_data_repo = request.json['source-data-repo']
    
    start = time.time()

    do_train(model_name=ML_model, 
        repo_url=source_data_repo, 
        repo_data_directory=source_data_directory, 
        data_format=data_format)
    
    end = time.time()

    runningtime = (end - start)

    return f'Model {ML_model} trained using source data in directory {source_data_directory}, running time {runningtime} seconds', 201

@app.route('/ccaas/api/v0.1/models', methods=['GET'])
@swag_from('models.yml')
def models():
    """
    List available models
    """
    models = get_models()
    print(f'Available models: {models}')
    return jsonify(models), 200

@app.route('/ccaas/api/v0.1/trained_models', methods=['GET'])
@swag_from('trained_models.yml')
def trained_models():
    """
    List trained models
    """
    trained_models = get_trained_models()
    print(f'Trained models: {trained_models}')
    return jsonify(trained_models), 200

def run():
    flask_run_host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    app.run(host=flask_run_host, debug=True) 

if __name__ == '__main__':
    run()