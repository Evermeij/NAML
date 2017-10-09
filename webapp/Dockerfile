FROM python:3.5-slim
#FROM continuumio/anaconda3:latest


RUN sed -i "s/httpredir.debian.org/debian.uchicago.edu/" /etc/apt/sources.list && \
    apt-get update && apt-get install -y build-essential
#RUN pip install wordcloud
ADD requirements.txt /
RUN pip install -r requirements.txt
COPY controller.py controller.py
COPY model.py model.py


ADD templates /templates
ADD static /static

ENTRYPOINT python controller.py
