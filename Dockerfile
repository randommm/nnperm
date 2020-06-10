# Base docker image
FROM pytorch/pytorch
MAINTAINER Marco Inacio <dockerfiles@marcoinacio.com>

RUN apt-get update && apt-get dist-upgrade -y && \
    apt-get install -y libpq-dev screen && \
    apt-get autoclean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN pip install sstudy peewee psycopg2-binary scipy nnlocallinear && \
    rm -rf /root/.cache/pip/

COPY . /opt
RUN cd /opt/package && python3 setup.py develop

CMD tail -f /dev/null
