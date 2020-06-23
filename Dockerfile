FROM python:3.8

COPY requirements.txt .
COPY app/ .
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3", "cli.py"]