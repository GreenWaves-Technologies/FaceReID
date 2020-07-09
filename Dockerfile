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
        imagemagick \
        rsync \
        autoconf \
        automake \
        texinfo \
        libtool \
        xxd \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install wheel && pip3 install opencv-python argcomplete six

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN mkdir -p /tmp/toolchain && cd /tmp/toolchain && \
    git clone https://github.com/GreenWaves-Technologies/gap_riscv_toolchain_ubuntu_18.git . && \
    ./install.sh /usr/lib/gap_riscv_toolchain && \
    rm -rf /tmp/toolchain

RUN mkdir /gap_sdk && cd /gap_sdk && \
      mkdir -p /root/.ssh/ && \
      ssh-keyscan github.com >> /root/.ssh/known_hosts && \
      git clone https://github.com/GreenWaves-Technologies/gap_sdk.git . && \
      git checkout release-v3.6 && \
      git submodule update --init --recursive && \
      pip3 install -r ./requirements.txt && \
      echo "https://greenwaves-technologies.com/autotiler/" > .tiler_url && \
      source ./configs/gapoc_a_v2.sh && make all && \
      TILER_LICENSE_AGREED=1 make autotiler && \
      source ./configs/gapuino_v2.sh && make all && \
      TILER_LICENSE_AGREED=1 make autotiler

RUN useradd ci -m -s /bin/bash -G users,dialout

USER ci

CMD bash
