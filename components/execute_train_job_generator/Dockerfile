# AI Platform Trainのカスタムコンテナ実行用にはgpu、jobキック用にはcpuを使用する
# CPU版："1.15.2-py3"、GPU版："1.15.2-gpu-py3"
ARG USE_IMAGE_TAG=1.15.2-py3
FROM tensorflow/tensorflow:${USE_IMAGE_TAG}

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    lsb-release \
    curl \
    wget

# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y


WORKDIR /pipelines/component/src/

# Setup Object Detection API
RUN git clone https://github.com/tensorflow/models.git

# Compile protobuf configs
WORKDIR /pipelines/component/src/models/research
RUN protoc object_detection/protos/*.proto --python_out=.

# ローカルの「setup.py」をコンテナにコピーする
COPY ./src/setup.py ./setup.py
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

# インストール実行（setup.pyに記載のライブラリがここでインストールされる）
RUN python -m pip install --user -U pip
RUN python -m pip install --user .
# 正常にインストールされたかチェック
RUN python object_detection/builders/model_builder_tf1_test.py

# setup oid pretrained model
RUN curl -OL http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz && \
    tar -zxvf ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz && rm ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz

COPY ./src .

# Trainプログラムのエントリーポイントを登録
# https://cloud.google.com/ai-platform/training/docs/using-containers?hl=ja#create_a_custom_container
ENTRYPOINT ["python", "object_detection/model_main.py"]

ENV TF_CPP_MIN_LOG_LEVEL 3