FROM python:3.8

COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY app/ .
COPY testdata/ ./testdata/
ENV MODEL_DIR /usr/app/mnt
ENV LOCAL_TRAIN_DATA_DIR /emission_data
ENV FLASK_RUN_HOST 0.0.0.0
EXPOSE 5000
ENTRYPOINT ["python3", "cli.py"]