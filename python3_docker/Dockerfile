FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y python3-dev python3-pip
RUN pip3 install --upgrade pip==9.0.1
RUN apt-get install -y python3-tk
RUN pip3 install open3d-python
RUN apt-get install -y vim
RUN pip3 install lmdb>=0.9
RUN pip3 install matplotlib>=2.1.0
RUN pip3 install msgpack==0.5.6
RUN pip3 install numpy>=1.14.0
RUN pip3 install pyarrow==0.10.0
RUN pip3 install tensorflow-gpu==1.12.0
RUN pip3 install tensorpack==0.8.9

EXPOSE 8888
WORKDIR /code