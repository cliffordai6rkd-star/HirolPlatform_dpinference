FROM 666860.xyz/arm64v8/ubuntu:20.04

RUN mkdir -p /root/.pip
COPY pip.conf /root/.pip/pip.conf
COPY sources.list /etc/apt/sources.list
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt update && \
    apt install -y \
    build-essential \
    zlib1g-dev \
    libffi-dev \
    libssl-dev \
    libsqlite3-dev \
    libreadline-dev \
    libncursesw5-dev \
    libgdbm-dev \
    libbz2-dev \
    liblzma-dev \
    tk-dev \
    libdb-dev \
    uuid-dev \
    xz-utils \
    wget \
    cmake \
    make

WORKDIR  /root
RUN wget https://mirrors.huaweicloud.com/python/3.8.10/Python-3.8.10.tar.xz && tar -xvf Python-3.8.10.tar.xz
WORKDIR  /root/Python-3.8.10
RUN ./configure --prefix=/usr/local && make -j8 && make install

RUN ln -s pip3 pip

WORKDIR /workspace/unitree_sdk2_python