from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import random

app = Flask(__name__)
api = Api(app)

co2e = random.randint(150,666)
ci = round(co2e/10)
mean = co2e - 2
median = co2e - 4

CO2E = {
    'CO2e': co2e,
    'confidence level': ci,
    'mean': mean,
    'median': median
}


# TodoList
# shows a list of all todos, and lets you POST to add new tasks
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
## Actually setup the Api resource routing here
##
api.add_resource(Predict, '/cloth/<string:cloth>')

if __name__ == '__main__':
    app.run(debug=True)