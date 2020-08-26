This folder is used by the Dockerized version of the app. This folder is mounted into the Docker container and is used for storing the trained model files.

Everything in this folder is ignored by git except this file, readme.txt.

If you want to use pretrained models, you can copy the model files from the folder /pretrained_models_textile-v1.0.0 into this folder. You can then predict using them (without repeating the training).