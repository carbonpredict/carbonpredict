# Run with python dummyapi.py
# Then try e.g. http://127.0.0.1:5000/cloth/t-shirt on local computer

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import random

app = Flask(__name__)
api = Api(app)


# Gives a random "prediction" for any cloth
# E.g. http://127.0.0.1:5000/cloth/t-shirt on local computer
class Predict(Resource):
    def get(self, cloth):
        co2e = random.randint(150,666)
        ci = round(co2e/10)
        mean = co2e - 2
        median = co2e - 4
        CO2E = {
            'CO2e': co2e,
            '95% confidence level': ci,
            'mean': mean,
            'median': median
            }
        return CO2E

##
## Seteup the Api resource routing
##
api.add_resource(Predict, '/cloth/<string:cloth>')

if __name__ == '__main__':
    app.run(debug=True)