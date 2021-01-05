FROM python:3.7

COPY Requirements.txt Requirements.txt 
RUN pip install -r ./Requirements.txt 

ENV APP_HOME /app   

ENV FLASK_APP=Messenger.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000    

COPY . /app

WORKDIR /app

EXPOSE 5000

CMD ["flask","run"]
