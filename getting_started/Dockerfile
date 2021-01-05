FROM python:3.7

COPY requirements.txt requirements.txt 
RUN pip install -r ./requirements.txt 

ENV APP_HOME /app        
#removed for dev mode

ENV FLASK_APP=myapp.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=3000
#set workspace
WORKDIR ${APP_HOME}      
#removed for dev mode

#copy local files
#COPY . .                 removed for dev mode mount volume to /app by (-v 'pwd':/app)

EXPOSE 3000

CMD ["flask","run"]