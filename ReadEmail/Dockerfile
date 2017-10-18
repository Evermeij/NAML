FROM    openjdk:8-alpine
RUN     mkdir -p /app
ADD     /ReadEmail-all-1.0-SNAPSHOT.jar /app/app.jar
WORKDIR /app
CMD     java -jar app.jar