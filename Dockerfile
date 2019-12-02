FROM ubuntu:18.04

USER root

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ssh \
        sudo \
        build-essential \
        make \
        gcc \
        g++ \
        git \
        git-lfs \
        libftdi-dev \
        libftdi1 \
        doxygen \
        python \
        python-pip \
        python3-pip \
        python3-setuptools \
        python3-dev \
        libsdl2-dev \
        curl \
        wget \
        rsync \
        cmake \
        libusb-1.0-0-dev \
        scons \
        gtkwave \
        libsndfile1-dev \
        imagemagick && \
        ln -s /usr/bin/libftdi-config /usr/bin/libftdi1-config && \
        ln -s /usr/lib/x86_64-linux-gnu/libmpfr.so.6 /usr/lib/x86_64-linux-gnu/libmpfr.so.4 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install wheel && pip3 install pyelftools opencv-python

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN mkdir -p /tmp/toolchain && cd /tmp/toolchain && \
    git lfs clone https://github.com/GreenWaves-Technologies/gap_riscv_toolchain.git . && \
    ./install.sh && \
    rm -rf /tmp/toolchain

ENV TARGET_CHIP="GAP8"
ENV TARGET_NAME="gap"
ENV PULP_CURRENT_CONFIG=gap_rev1@config_file=chips/gap/gap.json

RUN mkdir /gap_sdk && cd /gap_sdk && \
      mkdir -p /root/.ssh/ && \
      ssh-keyscan github.com >> /root/.ssh/known_hosts && \
      git clone https://github.com/GreenWaves-Technologies/gap_sdk.git . && \
      git checkout master && \
      git submodule update --init --recursive && \
      echo "https://greenwaves-technologies.com/autotiler/" > .tiler_url && \
      source ./sourceme.sh && make all autotiler

RUN useradd ci -m -s /bin/bash -G users,dialout

USER ci

CMD bash
