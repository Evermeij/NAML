FROM python:3.5-slim

RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get -y install tk-dev && rm -r /var/lib/apt/lists/*

ADD app/ app/
ADD java /java

WORKDIR /app
RUN pip install -U -r /app/requirements.txt
RUN python -m nltk.downloader stopwords
#ENTRYPOINT python app/controller.py
#CMD ["python","-u","app.py"]
