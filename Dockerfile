FROM hkleeeee/torch_segmentation:1.13.1-cuda11.6-cudnn8-devel
MAINTAINER Heekyoung Lee <dlapdlf1739@gmail.com>

RUN apt-get update && apt-get install git nano -y
# VOLUMNE

EXPOSE 6006
EXPOSE 8888

WORKDIR /
