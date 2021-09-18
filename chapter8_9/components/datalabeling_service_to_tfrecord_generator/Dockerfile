ARG BASE_IMAGE_TAG=3.7-slim
FROM python:$BASE_IMAGE_TAG

WORKDIR /pipelines/component/src/
COPY ./src/* ./
RUN pip install -U pip && pip install \
      -r requirements.txt