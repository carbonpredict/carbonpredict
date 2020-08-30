# Carbonpredict
Machine learning pipeline and model to predict CO2e of products

## What is it?
The application is a dockerized machine learning pipeline that can ingest a source data set, train models on it and then be able to predict CO2 emission values for clothing products.

## Repository contents
*[/app](/app)* contains the python source code for the main application. For instruction on running the app, see section *How to run it*. 

*[/notebooks](/notebooks)* contains Jupyter Notebooks used in experimenting with ML-models and doing visualizations the project. 

*[/pretrained_models_textile-v1.0.0](/pretrained_models_textile-v1.0.0)*. There models have been trained using the dataset *textile-v.1.0.0*.

*[/testdata](/testdata)* contains json test data to test the predict API with and csv test data to test the predict (from CLI with).

*[/visualizations](/visualizations)* contains some visualizations (png images) of the source data. These have been output from the notebook Visualizations.ipynb.

*Dockerfile*, *docker-compose.yml* and *requirements.txt* at the root contain the Docker definitions of the app. The folders *[/mnt_emission_data](/mnt_emission_data)* and *[/mnt_models](/mnt_models)* are used by the Dockerized version of the app for storing source emission data and storing the trained models (respectively). The folders are mounted into the container.

*README.md*, *predict_api.md* are general documentation.

## How to run it
The solution is dockerized. Install Docker on your machine, instructions at the [Docker get started section](https://docs.docker.com/get-started/).

### Build image
Currently the image is not offered through Dockerhub. Build the image locally by cloning this repository, navigating to the repo's root folder and running the command `docker build -t carbon .`. This builds the Docker image with the tag *carbon*. The other docker commands are run using this image.

### List models
To list the models available to be trained, run `docker-compose run carbon models`.

### List trained models
To list the trained models ready to be called on to predict CO2 equivalents, run `docker-compose run carbon trained_models`.

### Train model
To train a model (here *lgbm_default*, a gradient boosting model) with the source data, run `docker-compose run carbon train lgbm_default`. Substitute the model name as the last parameter in the command. This will mount your subdirectory */mnt_models* into the container, train a model and save it to the subdirectory.

As default, the train command clones the source data files (about 1 GB) into the container from a remote git repository. If you have the CSV-format source data files on your local computer, you can copy them into the subfolder */mnt_emission_data* and use the switch *--local_data* to use local data (from a docker-mounted directory) instead of cloning the data into the container. Example using the switch: `docker-compose run carbon train lgbm_default --local_data`

### Predict
To predict a CO2e value using a trained model, run a command like `docker-compose run carbon predict lgbm_default ./testdata/test.csv`, where the last two parameters are a trained model to use and the location of the csv file to do the predictions for. The columns of the CSV file must currently be in the exact order (and including the empty target column *co2_total*): brand,category-1,category-2,category-3,co2_total,colour,fabric_type,ftp_acrylic,ftp_cotton,ftp_elastane,ftp_linen,ftp_other,ftp_polyamide,ftp_polyester,ftp_polypropylene,ftp_silk,ftp_viscose,ftp_wool,gender,label,made_in,season,size,unspsc_code,weight

### Run server
To run the Flask server, which offers HTTP API endpoints (see list below), run `docker-compose run --service-ports carbon run-server`. 

After the server is started, you can test the API using the **OpenAPI UI** (a.k.a. Swagger UI) by going to http://*hostname*:5000/apidocs/ (when run locally http://localhost:5000/apidocs/ or http://0.0.0.0:5000/apidocs/).

## Current machine learning models

**Gradient boosting** (*lgbm_default* or *lgbm_qreg*):  Root mean squared error (RMSE) = 12.5, r2 (coefficient of determination) = 0.7935093722105526. Makes on average a 12,5 gram error on a single CO2e prediction. Quantile regression (*lgbm_qreg*) version provides additionally confidence intervals, while lgb default model (*lgbm_default*) tries to provide as accurate prediction as possible. 

**Neural network one-layer model** (*neural_onelayer*): RMSE = 9.822982788085938, R2 = 0.8707626750063729. A neural network model with one hidden layer (with 1024 neurons). Implemented using Pytorch. The model uses a GPU for training, evaluation and prediction is a GPU is available in the environment (note: when running in Docker container, you must make the GPU available within the container). Makes on average a 9,8 gram error on a single CO2e prediction. 

**Simple K-nearest neighbors** (*k_nearest_neighbors*) (using only category-1, category-2, category-3, fabric_type and size):  RMSE = 19.476921284404497, R2 = 0.5095385877102603. Makes on average a 19,5 gram error on a single CO2e prediction.

**Linear Regression** (*linear_reg*): RMSE = 15.45228099822998, R2 = 0.687436167942354. Uses all features except the empy features from the training data (*label* and *unspsc_code*). Makes an avarage 15,5 gram error on a single CO2e prediction.

### Pretrained models
There are pretrained models available in the subfolder *[/pretrained_models_textile-v1.0.0](/pretrained_models_textile-v1.0.0)*. There models have been trained using the dataset *textile-v.1.0.0*. If you want to use these models, you can copy the model files to the directory */mnt_models*. You can then use the models for predictions (without training new ones). Note: if you have copied a model file into the folder */mnt_models* and run the training of a model again, the model file in */mnt_models* will be overwritten.

## Adding new models
- Make your model implement the absract class CarbonModelBase defined in package models. Your model's *train* method must save your model into the path base_dir given as a parameter, and the *load* method must load your model from the given base_dir. The model file's name must start with the model's name followed by a hyphen (-) and use file extension .model (e.g. *k_nearest_neighbors-k_9_training_samples_100000.model*)
- Add your model files to a subdirectory under the package models
- Import and list your model in the list AVAILABLE_MODELS in models/\__init__.py
- Check that the packages you use are listed in the file *requirements.txt* and add packages if needed

## Used source data
The used source data is published in the [emission-sample-data repository](https://github.com/Compensate-Operations/emission-sample-data) by [Compensate](https://compensate.com/).

## APIs
All APIs documented also using OpenAPI (version 2), UI available at http://*hostname*:5000/apidocs/ (when run locally http://localhost:5000/apidocs/ or http://0.0.0.0:5000/apidocs/).

### Prediction API
See *[prediction API definition](predict_api.md)*.

As an alternative to using the OpenAPI UI, You can use CURL directly to send the product parameters to the predict endpoint in JSON format in an HTTP POST message. Example using curl (to be run inside directory */testdata*) using example data in file */testdata/wsocks.json*: `curl -i -H "Content-Type: application/json" -X POST --data "@wsocks.json" http://localhost:5000/ccaas/api/v0.1/predict`

### Models API
Endpoint: */ccaas/api/v0.1/models* (GET). Returns a JSON array of model names available to be trained. Example response body: ["k_nearest_neighbors", "lgbm_default", "neural_onelayer"].

### Trained models API
Endpoint: */ccaas/api/v0.1/trained_models* (GET). Returns a JSON array of names of trained models. Example response body: ["k_nearest_neighbors", "lgbm_default"].

### Train API
Endpoint: */ccaas/api/v0.1/train* (POST). HTTP body is type application/json and contains the training data and machine learning model definition in JSON format. NOTE: Training is a slow operation and might take e.g. 5-30 minutes depending on the model, source data size and host machine resources.

#### Example message body
```
{
    "data-format": "tgz",
    "ML-model": "lgbm_default",
    "source-data-directory": "datasets/textile-v1.0.0",
    "source-data-repo": "https://github.com/Compensate-Operations/emission-sample-data.git"
}
```

## Possible future features
- Use a more varied and more realistic dataset
- Add prediction intervals (5-percentile and 95-percentile) to more models
- Extend software to support other product categories

## Contributors
Project group:
- Mikko Kotola
- Jyrki Martti
- Petteri Muilu
- Ville-Veikko Saari

## Random
*Carbon predict* organization profile image: CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=2099935
