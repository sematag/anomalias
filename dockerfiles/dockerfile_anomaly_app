FROM facundogp/anom-api:v1-py3.9

RUN apt update && apt-get install -y git

RUN ln -s /usr/local/bin/python3.9 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.9 /usr/bin/pip && \
    pip install cython 

RUN git clone --recursive http://github.com/ibayer/fastFM.git

WORKDIR ./fastFM

RUN pip install -r ./requirements.txt && \
    make TARGET=NEHALEM && \
    python setup.py install

WORKDIR /

RUN rm -rf fastFM/
