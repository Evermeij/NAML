FROM python:3.5-slim

RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get -y install tk-dev && rm -r /var/lib/apt/lists/*

ADD /app/requirements.txt /app/requirements.txt

RUN pip install -U -r /app/requirements.txt
RUN python -m nltk.downloader stopwords

ADD app/ app/

WORKDIR /app
#add expose for discipl
#EXPOSE 80

#CMD ["python","-u","app.py"]
CMD ["gunicorn", "--config=/app/gunicorn.py", "app:app"]

#ENTRYPOINT python app/controller.py
#CMD ["python","-u","app.py"]

#command: python -u app.py
#command: gunicorn app:app -w 8 -b  0.0.0.0:5000 --name app --log-level=debug --log-file=- #-b   0.0.0.0:5000 or :8000

