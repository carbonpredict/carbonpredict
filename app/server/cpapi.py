#!flask/bin/python
from flask import Flask, jsonify, abort, make_response, request, url_for
from werkzeug.utils import secure_filename
from cli import do_train, do_prediction_with_params, load_model
from flasgger import Swagger, swag_from
import random
import os

"""
This is an initial inplementtion of a REST based API and server for the CCaaS service.
As input the following json format with example values is expected. 
At least the "category-3" feature needs to have a value. Other 
features are optional. Also the ML used in prediction can be selected.
The default ML-model is "xyz".
{
    "brand": "b83",
    "category-1": "womenswear",
    "category-2": "footwear",
    "category-3": "socks",
    "colour": "bondi blue",
    "fabric_type": "K",
    "ftp_acrylic": "21.0",
    "ftp_cotton": "",
    "ftp_elastane": "",
    "ftp_linen": "",
    "ftp_other": "",
    "ftp_polyamide": "43.0",
    "ftp_polyester": "",
    "ftp_polypropylene": "24.0",
    "ftp_silk": "",
    "ftp_viscose": "",
    "ftp_wool": "11.0",
    "gender": "W",
    "label": "",
    "made_in": "VN",
    "season": "",
    "size": "M",
    "unspsc_code": "",
    "weight": "0.029",
    "ML-model": "", 
}

Example usage. 1. Run the server and the api. 2. Call it with curl using wsocks.json file with test input
$ python cpapi.py
$ curl -i -H "Content-Type: application/json" -X POST --data "@wsocks.json" http://localhost:5000/ccaas/api/v0.1/predict
"""

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'json'}

app = Flask(__name__)

app.config['SWAGGER'] = {
    'title': 'Carbon predict API',
    'url': '/ccaas/api/v0.1',
    'version': "0.1",
    'uiversion': 3
}

swagger = Swagger(app)

@app.route('/ccaas/api/v0.1/predict', methods=['POST'])
@swag_from('predict.yml')
def predict():    
    if not request.json or not 'category-3' in request.json:
        abort(400)
    
    product = {
    "brand": request.json["brand"],
    "category-1": request.json["category-1"],
    "category-2": request.json["category-2"],
    "category-3": request.json["category-3"],
    "colour": request.json["colour"],
    "fabric_type": request.json["fabric_type"],
    "ftp_acrylic": request.json["ftp_acrylic"],
    "ftp_cotton": request.json["ftp_cotton"],
    "ftp_elastane": request.json["ftp_elastane"],
    "ftp_linen": request.json["ftp_linen"],
    "ftp_other": request.json["ftp_other"],
    "ftp_polyamide": request.json["ftp_polyamide"],
    "ftp_polyester": request.json["ftp_polyester"],
    "ftp_polypropylene": request.json["ftp_polypropylene"],
    "ftp_silk": request.json["ftp_silk"],
    "ftp_viscose": request.json["ftp_viscose"],
    "ftp_wool": request.json["ftp_wool"],
    "gender": request.json["gender"],
    "label": request.json["label"],
    "made_in": request.json["made_in"],
    "season": request.json["season"],
    "size": request.json["size"],
    "unspsc_code": request.json["unspsc_code"],
    "weight": request.json["weight"],
    "ML-model": request.json["ML-model"]
    }

    product["co2_total"] = None
    ml_model = product.pop("ML-model", None)
    if (ml_model == None or ml_model == ""):
        print('Loading default model: LGBM')
        ml_model = 'lgbm_default'
    else:
        print(f'Loading model: {ml_model}')
    model = load_model(ml_model)
    print('Model loaded')
    CO2e = do_prediction_with_params(model, product)
    print('CO2e prediction complete, returning result')
    print(CO2e)
    return CO2e, 201

def run():
    flask_run_host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    app.run(host=flask_run_host, debug=True) 

if __name__ == '__main__':
    run()