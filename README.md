# Carbonpredict
Machine learning pipeline and model to predict CO2e of products

## What is it?
*Goal for project ending 8/2020*: The application is a machine learning pipeline that can ingest a source data set, train a model on it and then be able to predict CO2 emission values for clothing products.

## Repository contents
*Notebooks* contains Jupyter Notebooks used in the project. 

For loading the source data, there are two approaches in the notebooks:
- Some notebooks (e.g. Clustering.ipynb) rely on the source data being in the folder */data* under the repository root (data is not uploaded to GitHub). This is handy if you are running the notebook locally and already have downloaded and unzipped the source data.
- Some notebooks (e.g. Carbon_EDA_1.ipynb) clone the source data git repository (into a subfolder /content ) and unzip the source data there. This is handy if you are running the notebook in Colab.

*Visualizations* contains some visualizations (png images) of the source data. These have been output from the notebook Visualizations.ipynb.

*App* contains the main application (template at first).

*Dockerfile* at the root contains the definition of the container for Docker.

## How to run it
The solution is dockerized. Install Docker on your machine, instructions at the [Docker get started section](https://docs.docker.com/get-started/).

### Build image
Currently the image is not offered through Dockerhub. Build the image locally by cloning this repository, navigating to the repo's root folder and running the command `docker build -t carbonpred .`. This builds the Docker image with the tag *carbonpred*. The other docker commands are run using this image.

### List models
To list the available models, run `docker run carbonpred models`.

### Train model
To train a model (here *lgbm_default*, a gradient boosting model) with the source data, run `docker run carbonpred train lgbm_default`. Substitute the model name as the last parameter in the command.

### Predict
To predict a CO2e value using a trained model, run `docker run carbonpred predict`. *Predictions through Docker has not been implemented yet.*

### Run server
To run the demo server, which offers an HTTP endpoint for calling the predict command, run `docker run carbonpred run-server`. *Running the server through Docker has not been implemented yet.*

## Used source data
The used source data is published in the [emission-sample-data repository](https://github.com/Compensate-Operations/emission-sample-data) by [Compensate](https://compensate.com/).

## APIs
### Prediction API
See *[prediction API definition](predict_api.md)*.

### Admin API
To be defined.

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
