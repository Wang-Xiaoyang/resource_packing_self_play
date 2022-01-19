FROM nvidia/cuda:10.2-base

MAINTAINER xw17070 "xw17070@bristol.ac.uk"

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3.6
RUN apt-get -y install python3-pip

RUN apt-get install -y gcc

RUN pip3 install --upgrade wheel

RUN pip3 install setuptools --upgrade

RUN pip3 install -r requirements.txt

ENV WANDB_API_KEY 418b945aa8341019191bc5deb7539c50ffb9bb88
#ENV CUDA_VISIBLE_DEVICES 0

#COPY . /

#ENTRYPOINT [ "python3" ]

#CMD ["xw_mcts/main_bpp.py"]
