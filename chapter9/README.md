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
$ export PROJECT_ID=ai-shoseki-ai-platform
$ gcloud config set project ${PROJECT_ID}
$ gcloud beta auth application-default login
```

環境変数 `CLOUDSDK_CONFIG` が適切に設定されていること

```shell script
$ export CLOUDSDK_CONFIG=$HOME/.config/gcloud/
```

## 2. コンテナのビルド

コンポーネントと処理概要は以下の通り。コンテナは、GCR（Google Container Registry）で管理する

| コンポーネント名| 概要 | GCR|
| :- | :------------------------------------------------------------------------------- | :--------------- |
| datalabeling_service_to_<br> tfrecord_generator | データラベリングサービスのアノテーションデータをtfrecord形式に変換する| gcr.io/ai-shoseki-ai-platform/datalabeling_service_to_tfrecord_generator:latest|
| execute_train_job_<br>generator | Object Detection APIの物体検出モデルをAI Platform上で学習させるジョブを実行する（CPU版。ローカルで利用するコンテナ）  | gcr.io/ai-shoseki-ai-platform/execute_train_job_generator:latest-1.15.2-py3|
| execute_train_job_<br>generator| Object Detection APIの物体検出モデルをAI Platform上で学習させるジョブを実行する（GPU版。AI Platform Trainで利用するカスタムコンテナ用） | gcr.io/ai-shoseki-ai-platform/execute_train_job_generator:latest-1.15.2-gpu-py3|
| model_compile_generator| クラウド or エッジの推論用にモデルをコンパイルする |gcr.io/ai-shoseki-ai-platform/model_compile_generator:latest-1.15.2-py3|

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
    -t gcr.io/ai-shoseki-ai-platform/datalabeling_service_to_tfrecord_generator:latest \
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
    -t gcr.io/ai-shoseki-ai-platform/execute_train_job_generator:latest-1.15.2-py3 \
    python3 /work/src/main.py \
    --cloud_yaml_config_path=/work/src/cloud.yaml \
    --num_train_steps=100 \
    --model_dir=gs://shoseki-ai-platform/train/object_detection_train_20210307112411
```

### 3.3. model_compile_generator
```shell script
$ cd components/model_compile_generator
$ docker run --rm -it \
    --volume $(pwd):/work \
    --volume ${CLOUDSDK_CONFIG}:/credentials \
    --env CLOUDSDK_CONFIG=/credentials \
    --entrypoint "" \
    -t gcr.io/ai-shoseki-ai-platform/model_compile_generator:latest-1.15.2-py3 \
    python3 /work/src/main.py
```

## 4. Pipeline のコンパイル

```shell script
dsl-compile --py pipelines/tests/datalabeling_service_to_tfrecord_generator_pipeline.py --output pipelines/tests/datalabeling_service_to_tfrecord_generator.tar.gz
```
