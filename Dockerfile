FROM python:3.5-slim
#FROM continuumio/anaconda3:latest

RUN apt-get update && \
    apt-get -y install gcc # mono-mcs # && \
#    rm -rf /var/lib/apt/lists/*
#RUN sed -i "s/httpredir.debian.org/debian.uchicago.edu/" /etc/apt/sources.list && \
#    apt-get update && apt-get install -y build-essential

ADD machine_learning/ machine_learning/
ADD webapp/ webapp/
RUN ls -la /webapp/*
#ADD templates webapp/templates/
#ADD static webapp/static/

RUN pip install -r webapp/requirements.txt

ENTRYPOINT python webapp/controller.py
