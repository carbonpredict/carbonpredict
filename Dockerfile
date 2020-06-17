FROM python:3.8

COPY app/ .

ENTRYPOINT ["python3", "cli.py"]