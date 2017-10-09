FROM python:3.6-slim

ADD requirements.txt /
RUN pip install -r requirements.txt

COPY database.py database.py

ENTRYPOINT python database.py
