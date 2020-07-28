FROM python:3.8

COPY requirements.txt .
COPY app/ .
COPY testdata/ ./testdata/
RUN pip3 install -r requirements.txt
ENV MNT_DIR /usr/app/mnt
ENV FLASK_RUN_HOST 0.0.0.0
EXPOSE 5000
ENTRYPOINT ["python3", "cli.py"]