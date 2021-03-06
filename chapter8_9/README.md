# aiplatform-pipeliness

## 1. 前準備

### 1.1. EnableにしておくべきAPI

* Dataflow
* Google Container Registry
* Cloud Build
* AI Platform
* Data Labeling Service


### 1.2. 認証

```
$ export PROJECT_ID=${your-gcp-project}
$ gcloud config set project ${PROJECT_ID}
$ export CLOUDSDK_CONFIG=$HOME/.config/gcloud/
$ gcloud beta auth application-default login
$ gcloud auth login
```


## 2. コンテナのビルド

コンポーネントと処理概要は以下の通り。コンテナは、GCR（Google Container Registry）で管理する

| コンポーネント名| 概要 | GCR|
| :- | :------------------------------------------------------------------------------- | :--------------- |
| datalabeling_service_to_<br> tfrecord_generator | データラベリングサービスのアノテーションデータをtfrecord形式に変換する| gcr.io/${your-gcp-project}/datalabeling_service_to_tfrecord_generator:latest|
| execute_train_job_<br>generator | Object Detection APIの物体検出モデルをAI Platform上で学習させるジョブを実行する（CPU版。ローカルで利用するコンテナ）  | gcr.io/${your-gcp-project}/execute_train_job_generator:latest-1.15.2-py3|
| execute_train_job_<br>generator| Object Detection APIの物体検出モデルをAI Platform上で学習させるジョブを実行する（GPU版。AI Platform Trainで利用するカスタムコンテナ用） | gcr.io/${your-gcp-project}/execute_train_job_generator:latest-1.15.2-gpu-py3|
| model_compile_generator| クラウド or エッジの推論用にモデルをコンパイルする |gcr.io/${your-gcp-project}/model_compile_generator:latest-1.15.2-py3|

Cloud Buildを使用して、上記4つのコンテナを作成するジョブを実行する

```shell script
# Cloud Buildを使用したクラウド上でのDocker イメージ作成
$ cd components
$ gcloud builds submit --config=cloudbuild.yaml
```

## 3. ローカルからジョブを実行

### 3.1. datalabeling_service_to_tfrecord_generator
```shell script
$ cd components/datalabeling_service_to_tfrecord_generator
$ docker run --rm -it \
    --volume $(pwd):/work \
    --volume ${CLOUDSDK_CONFIG}:/credentials \
    --env CLOUDSDK_CONFIG=/credentials \
    --entrypoint "" \
    -t gcr.io/${your-gcp-project}/datalabeling_service_to_tfrecord_generator:latest \
    python3 /work/src/main.py

```

### 3.2. execute_train_job_generator
```shell script
$ cd components/executetrain_job_generator
$ docker run --rm -it \
    --volume $(pwd):/work \
    --volume ${CLOUDSDK_CONFIG}:/credentials \
    --env CLOUDSDK_CONFIG=/credentials \
    --entrypoint "" \
    -t gcr.io/${your-gcp-project}/execute_train_job_generator:latest-1.15.2-py3 \
    python3 /work/src/main.py \
    --cloud_yaml_config_path=/work/src/cloud.yaml \
    --num_train_steps=100
```

### 3.3. model_compile_generator
```shell script
$ cd components/model_compile_generator
$ docker run --rm -it \
    --volume $(pwd):/work \
    --volume ${CLOUDSDK_CONFIG}:/credentials \
    --env CLOUDSDK_CONFIG=/credentials \
    --entrypoint "" \
    -t gcr.io/${your-gcp-project}/model_compile_generator:latest-1.15.2-py3 \
    python3 /work/src/main.py
```

## 4. Pipeline のコンパイル

```shell script
dsl-compile --py pipelines/tests/datalabeling_service_to_tfrecord_generator_pipeline.py --output pipelines/tests/datalabeling_service_to_tfrecord_generator.tar.gz
```
