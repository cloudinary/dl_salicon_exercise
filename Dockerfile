ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3
FROM $BASE_IMAGE AS build

# install model server requirements
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt



