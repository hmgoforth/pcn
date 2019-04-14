FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
ENV TENSORFLOW_VERSION=1.4.1

RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
      build-essential \
      cmake \
      git \
      curl \
      vim \
      wget \
      ca-certificates \
      libjpeg-dev \
      libpng-dev  \
      openssl \
      libssl-dev \
      python-setuptools \
      unzip \
      net-tools\
      libgl1-mesa-glx

RUN easy_install pip
#RUN wget https://www.python.org/ftp/python/2.7.10/Python-2.7.10.tgz
#RUN tar -xvf Python-2.7.10.tgz

#RUN cd Python-2.7.10 && ./configure && make && make install

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
     libgtk2.0-dev \
     x11-xserver-utils \
     python-dev \
     python-tk

RUN pip install --upgrade pip

# Install TensorFlow and Keras
RUN pip install tensorflow-gpu==${TENSORFLOW_VERSION} keras h5py

ENV PATH=/usr/local/nvidia/bin:$PATH

RUN apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
    ethtool \
    less \
    iproute
RUN pip install --upgrade setuptools

RUN pip install awscli
RUN pip install boto3
RUN pip install opencv-python
RUN pip install Cython
RUN pip install matplotlib
RUN git clone https://github.com/tensorpack/tensorpack.git /tensorpack
RUN pip install --ignore-installed -e /tensorpack
RUN pip install numpy
RUN pip install pyarrow==0.10.0
RUN pip install lmdb
RUN pip install open3d-python
RUN pip install git+https://github.com/daavoo/pyntcloud
RUN pip install jupyter
EXPOSE 8888
WORKDIR /code

