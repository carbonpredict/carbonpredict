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
Currently the image is not offered through Dockerhub. Build the image locally by cloning this repository, navigating to the repo's root folder and running the command `docker build -t carbon .`. This builds the Docker image with the tag *carbon*. The other docker commands are run using this image.

### List models
To list the available models, run `docker-compose run carbon models`.

### Train model
To train a model (here *lgbm_default*, a gradient boosting model) with the source data, first create a subdirectory */mnt* at the root of the repository and then run run `docker-compose run carbon train lgbm_default`. Substitute the model name as the last parameter in the command. This will mount your subdirectory */mnt* into the container, train a model and save it to the subdirectory.

### Predict
To predict a CO2e value using a trained model, run a command like `docker-compose run carbon predict lgbm_default ./testdata/test.csv`, where the last two parameters are a trained model to use and the location of the csv file to do the predictions for. The columns of the CSV file must currently be in the exact order (and including the empty target column *co2_total*):
brand,category-1,category-2,category-3,co2_total,colour,fabric_type,ftp_acrylic,ftp_cotton,ftp_elastane,ftp_linen,ftp_other,ftp_polyamide,ftp_polyester,ftp_polypropylene,ftp_silk,ftp_viscose,ftp_wool,gender,label,made_in,season,size,unspsc_code,weight

### Run server
To run the demo server, which offers an HTTP endpoint for calling the predict command, run `docker-compose run --service-ports carbon run-server`. *The current implementation returns a randomly generated response value. To be done: making the server call the predict function.*

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
