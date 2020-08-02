FROM python:3.8

COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY app/ .
COPY testdata/ ./testdata/
ENV MNT_DIR /usr/app/mnt
ENTRYPOINT ["python3", "cli.py"]