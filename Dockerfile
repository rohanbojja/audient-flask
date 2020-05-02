FROM python:3

LABEL MAINTANER Your Name "rohanbojja@outlook.com"

RUN apt-get update -y && \
    apt-get -y install libsndfile1 ffmpeg

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -U pip

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python" ]

EXPOSE 5000

CMD [ "app.py" ]
