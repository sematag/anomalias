FROM debian:bullseye

RUN apt update

RUN apt install -y wget \
	 	   build-essential \
		   libncurses5-dev \
		   zlib1g-dev \
		   libnss3-dev \
		   libgdbm-dev \
		   libssl-dev \
		   libsqlite3-dev \
		   libffi-dev \
		   libreadline-dev \
		   curl \
		   libbz2-dev

RUN wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz

RUN tar -xvf Python-3.9.7.tgz

RUN rm Python-3.9.7.tgz

WORKDIR ./Python-3.9.7

RUN ./configure --enable-optimizations

RUN make -j $(nproc)

RUN make altinstall

WORKDIR /

RUN rm -rf ./Python-3.9.7

