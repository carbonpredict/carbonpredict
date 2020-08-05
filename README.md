# Carbonpredict
Machine learning pipeline and model to predict CO2e of products

## What is it?
*Goal for project ending 8/2020*: The application is a machine learning pipeline that can ingest a source data set, train a model on it and then be able to predict CO2 emission values for clothing products.

## Repository contents
*App* contains the python source code for the main application. For instruction on running the app, see section *How to run it*. 

*Notebooks* contains Jupyter Notebooks used in experimenting with ML-models and doing visualizations the project. 

For loading the source data, there are two approaches in the notebooks:
- Some notebooks (e.g. Clustering.ipynb) rely on the source data being in the folder */data* under the repository root (data is not uploaded to GitHub). This is handy if you are running the notebook locally and already have downloaded and unzipped the source data.
- Some notebooks (e.g. Carbon_EDA_1.ipynb) clone the source data git repository (into a subfolder /content ) and unzip the source data there. This is handy if you are running the notebook in Colab.

*Visualizations* contains some visualizations (png images) of the source data. These have been output from the notebook Visualizations.ipynb.

*Dockerfile*, *docker-compose.yml* and *requirements.txt* at the root contain the Docker definitions of the app. The folders */emission_data* and */mnt* are used by the Dockerized version of the app for storing source emission data and storing the trained models (respectively). The folders are mounted into the container.

*README.md*, *predict_api.md* are general documentation.

## How to run it
The solution is dockerized. Install Docker on your machine, instructions at the [Docker get started section](https://docs.docker.com/get-started/).

### Build image
Currently the image is not offered through Dockerhub. Build the image locally by cloning this repository, navigating to the repo's root folder and running the command `docker build -t carbon .`. This builds the Docker image with the tag *carbon*. The other docker commands are run using this image.

### List models
To list the available models, run `docker-compose run carbon models`.

### Train model
To train a model (here *lgbm_default*, a gradient boosting model) with the source data, run `docker-compose run carbon train lgbm_default`. Substitute the model name as the last parameter in the command. This will mount your subdirectory */mnt_models* into the container, train a model and save it to the subdirectory.

As default, the train command clones the source data files (about 1 GB) into the container from a remote git repository. If you have the CSV-format source data files on your local computer, you can copy them into the subfolder */mnt_emission_data* and use the switch *--local_data* to use local data (from a docker-mounted directory) instead of cloning the data into the container. Example using the switch: `docker-compose run carbon train lgbm_default --local_data`

### Predict
To predict a CO2e value using a trained model, run a command like `docker-compose run carbon predict lgbm_default ./testdata/test.csv`, where the last two parameters are a trained model to use and the location of the csv file to do the predictions for. The columns of the CSV file must currently be in the exact order (and including the empty target column *co2_total*):
brand,category-1,category-2,category-3,co2_total,colour,fabric_type,ftp_acrylic,ftp_cotton,ftp_elastane,ftp_linen,ftp_other,ftp_polyamide,ftp_polyester,ftp_polypropylene,ftp_silk,ftp_viscose,ftp_wool,gender,label,made_in,season,size,unspsc_code,weight

### Run server
To run the demo server, which offers an HTTP endpoint for calling the predict command, run `docker-compose run --service-ports carbon run-server`. Then, you can send the product parameters in JSON format in an HTTP POST message. Example using curl (to be run inside directory */testdata*) using example data in file */testdata/wsocks.json*: `curl -i -H "Content-Type: application/json" -X POST --data "@wsocks.json" http://localhost:5000/ccaas/api/v0.1/predict`   

## Current machine learning models

**Gradient boosting** (*lgbm_default*):  Root mean squared error (RMSE) = 12.5, r2 (coefficient of determination) = 0.7935093722105526. Makes on average a 12,5 gram error on a single CO2e prediction.

**Simple K-nearest neighbors** (*k_nearest_neighbors*) (using only 3 categories, fabric_type & size):  RMSE = 19.476921284404497, R2 = 0.5095385877102603. Makes on average a 19,5 gram error on a single CO2e prediction

## Used source data
The used source data is published in the [emission-sample-data repository](https://github.com/Compensate-Operations/emission-sample-data) by [Compensate](https://compensate.com/).

## APIs
### Prediction API
See *[prediction API definition](predict_api.md)*.

### Admin API
To be defined. Admin API will be used for training selected models on selected source data.

## Possible future features
- Extend software to support other product categories

## Contributors
Project group:
- Mikko Kotola
- Jyrki Martti
- Petteri Muilu
- Ville-Veikko Saari

## Random
*Carbon predict* organization profile image: CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=2099935
