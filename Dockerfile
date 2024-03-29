FROM python:3.10.6-buster


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ddd ddd
COPY setup.py setup.py
RUN pip install .
COPY training_outputs training_outputs



CMD uvicorn ddd.api.fast:app --host 0.0.0.0 --port $PORT
