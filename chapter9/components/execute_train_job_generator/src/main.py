import argparse
import datetime
import glob
import logging
import os
import sys
import time

import tensorflow as tf
from google.protobuf import text_format
from googleapiclient import discovery
from googleapiclient import errors
from object_detection.protos import pipeline_pb2


class ExecuteTrainJobGenerator:


    def __init__(self, args):
        """
        コンストラクタ

        Args:
            args (List[str]): コマンドライン引数
        """

        # ロガーの設定
        self._logger = self._get_logger()
        self._logger.info(f"TensorFlow version : {tf.__version__}")

        parser = argparse.ArgumentParser(
            description="launch execute train job generator")

        # GCPプロジェクトID
        parser.add_argument("--project_id", type=str, default="ai-shoseki-ai-platform",
                            help='GCP project id')

        # AI Platformのジョブを実行するリージョン
        parser.add_argument("--ai_platform_job_region", type=str, default="us-west1",
                            help='AI Platform train job execute region.')

        # AI Platformのジョブスクリプト格納用フォルダ
        parser.add_argument("--ai_platform_job_dir_path", type=str,
                            default="gs://shoseki-ai-platform",
                            help='AI Platform job files dir.')

        # AI Platformのジョブ名
        parser.add_argument("--job_name", type=str, default="object_detection_train",
                            help='AI Platform train job name. You can restart the '
                                 'checkpoint once you specify the previous job '
                                 'name. Note that object detection api pipeline config '
                                 'must be the same as last time.')

        # AI Platformのモデルパス
        parser.add_argument("--model_dir", type=str, default=None,
                            help='Path to the model being trained by the user.')

        # AI Platform Trainで使うマシンタイプ等の設定情報
        parser.add_argument("--cloud_yaml_config_path", type=str,
                            default="object_detection/samples/cloud/cloud.yml",
                            help='AI Platform training settings (machine '
                                 'specifications, number of masters and workers, etc.)')

        # AI Platform Trainで使うカスタムコンテナのパス
        parser.add_argument("--image_uri", type=str,
                            default="gcr.io/ai-shoseki-ai-platform/"
                                    "execute_train_job_generator:latest-1.15.2-gpu-py3",
                            help='AI Platform training custom container image uri on '
                                 'Container Registry.')

        # Object Detection API 関連
        # 学習させるクラス数
        parser.add_argument("--num_classes", type=int, default=2,
                            help='Number of classes (Train and validation are same) .')

        # ラベルマップのパス
        parser.add_argument("--label_map_path", type=str,
                            default="gs://shoseki-ai-platform/tfrecord/image"
                                    "/20200411152510/train/label_map.pbtxt",
                            help='Label map path (`pbtxt` format).')

        # tfrecord（学習用）のパス
        parser.add_argument("--tf_record_input_path_train", type=str,
                            default="gs://shoseki-ai-platform/tfrecord/image"
                                    "/20200411152510/train/data-00000-of-00001.tfrecord",
                            help='Tf record files path in GCS. For Train.')

        # tfrecord（評価用）のパス
        parser.add_argument("--tf_record_input_path_eval", type=str,
                            default="gs://shoseki-ai-platform/tfrecord/image"
                                    "/20200411152510/validation/data-00000-of-00001"
                                    ".tfrecord",
                            help='Tf record files path in GCS. For Evaluation.')

        # 学習時のバッチサイズ
        parser.add_argument("--train_config_batch_size", type=int, default=12,
                            help='Training batch size. The larger the batch size, '
                                 'the faster the training. However, '
                                 'it is easy to run out of memory.')
        # 評価に使うサンプル数
        parser.add_argument("--eval_config_num_examples", type=int, default=100,
                            help='Number of samples for evaluation.')

        # 学習のステップ数
        parser.add_argument("--num_train_steps", type=int, default=100,
                            help="Set train steps. If more train needs, up this param. "
                                 "Default 100 is for testing, very small.")

        # 事前学習済みモデルの名前
        parser.add_argument("--pre_trained_model_name", type=str,
                            default="ssd_mobilenet_v2_oid_v4_2018_12_12",
                            help='The name of the pretrained model.')

        # 量子化するかどうか
        parser.add_argument("--quantization", type=bool, default=False,
                            help='Whether to perform int8 quantization')

        # パイプラインconfigのパス
        parser.add_argument("--config_file_path", type=str,
                            help='Path for Configuration of Model.')

        # チェックポイントファイルのパス
        parser.add_argument("--checkpoints_path", type=str,
                            help='Path where check points are.')

        # コマンドライン引数をパースする
        self._args = parser.parse_args(args)
        self._logger.info(f'args: {self._args}')

        # 出力リソースに付加するバージョン情報
        self._output_version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


    def _get_logger(self):
        """
        Loggerオブジェクトを取得する

        Returns: Loggerオブジェクト
        """

        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return logger


    def _copy_resource(self, from_path, dest_dir):
        """
        プログラムリソースをコピーする

        Args:
            from_path (str): ローカルパス
            dest_dir (str): 宛先ディレクトリ

        """
        for local_file in glob.glob(from_path + '/**'):
            if not os.path.isfile(local_file):
                continue
            dest_file = os.path.join(dest_dir, local_file)
            tf.io.gfile.copy(local_file, dest_file, overwrite=True)
            self._logger.info(f'Copy from {local_file} to {dest_file}')


    def _setup_object_detection_pipeline_config(
        self,
        pre_trained_model_name,
        ai_platform_job_dir_path,
        num_classes,
        label_map_path,
        job_name,
        tf_record_input_path_train,
        tf_record_input_path_eval,
        eval_config_num_examples,
        train_config_batch_size,
        quantization
    ):
        """
        Object Detection API の PipelineConfig を作成する

        Args:
            pre_trained_model_name (str): 事前学習済みモデルの名前
            ai_platform_job_dir_path (str): AI Platformのジョブスクリプト格納用フォルダ
            num_classes (str): 学習させるクラス数
            label_map_path (str): ラベルマップのパス
            job_name (str): AI Platformのジョブ名
            tf_record_input_path_train (str): tfrecord（学習用）のパス
            tf_record_input_path_eval (str): tfrecord（評価用）のパス
            eval_config_num_examples (int): 評価に使うサンプル数
            train_config_batch_size (int): 学習時のバッチサイズ
            quantization (bool): 量子化するかどうか

        """

        pipeline_config_path = f"{pre_trained_model_name}/pipeline.config"
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

        with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)

        # class数に応じて設定
        pipeline_config.model.ssd.num_classes = num_classes

        pipeline_config.train_input_reader.label_map_path = label_map_path
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] \
            = tf_record_input_path_train
        pipeline_config.eval_input_reader[0].label_map_path \
            = label_map_path
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] \
            = tf_record_input_path_eval
        pipeline_config.train_config.fine_tune_checkpoint \
            = f"{ai_platform_job_dir_path}/train/" \
              f"{job_name}_{self._output_version}/" \
              f"{pre_trained_model_name}/model.ckpt"

        # デフォルトの評価メトリクス削除
        pipeline_config.eval_config.ClearField("metrics_set")
        pipeline_config.eval_config.ClearField("use_moving_averages")

        # 評価に使うサンプル数を指定
        pipeline_config.eval_config.num_examples = eval_config_num_examples

        # メモリサイズに応じてバッチサイズ設定
        # (バッチサイズ大きいと学習は速いけどメモリーオーバーになりやすい)
        pipeline_config.train_config.batch_size = train_config_batch_size

        config_text = text_format.MessageToString(pipeline_config,
                                                  double_format='.15g')

        new_pipeline_config_path = f"{pre_trained_model_name}/pipeline_new.config"
        with tf.io.gfile.GFile(new_pipeline_config_path, "wb") as f:
            f.write(config_text)
            if quantization:
                # 量子化の設定（int8）
                f.write("graph_rewriter { quantization "
                        "{ delay: 48000 weight_bits: 8 activation_bits: 8 } }")


    def _execute_ai_platform_job(
        self,
        job_name,
        ai_platform_job_dir_path,
        pre_trained_model_name,
        ai_platform_job_region,
        cloud_yaml_config_path,
        num_train_steps,
        config_file_path,
        checkpoints_path,
        image_uri,
        model_dir
    ):
        """
        AI Platformのジョブを実行する

        Args:
            job_name (str):
            ai_platform_job_dir_path (str): AI Platformのジョブスクリプト格納用フォルダ
            pre_trained_model_name (str): 事前学習済みモデルの名前
            ai_platform_job_region (str): AI Platformのジョブを実行するリージョン
            cloud_yaml_config_path (str): AI Platform Trainで使うマシンタイプ等の設定情報
            num_train_steps (int): 学習のステップ数
            config_file_path (str): パイプラインconfigのパス
            checkpoints_path (str): チェックポイントファイルのパス

        Returns:

        """
        model_dir_ = model_dir if model_dir else \
            f"{ai_platform_job_dir_path}/train/{job_name}_{self._output_version}"
        pipeline_config_path_ = f"{ai_platform_job_dir_path}/train/" \
                                f"{job_name}_{self._output_version}/" \
                                f"{pre_trained_model_name}/pipeline_new.config"

        cmd = f"""gcloud ai-platform jobs submit training 
        {job_name}_{self._output_version} \
            --region={ai_platform_job_region} \
            --config={cloud_yaml_config_path} \
            --master-image-uri={image_uri} \
            -- \
            --model_dir={model_dir_} \
            --num_train_steps={num_train_steps} \
            --pipeline_config_path={pipeline_config_path_}
        """

        self._logger.info(f'execute_ai_platform_job : {cmd}')

        try:
            # kubeflow pipeline実行時のメタパラメータ（コンポーネント間連携用）格納用
            # Create directory for storing output path
            os.makedirs(os.path.dirname(config_file_path))
            os.makedirs(os.path.dirname(checkpoints_path))

            with open(config_file_path, 'w') as f:
                f.write(pipeline_config_path_)
            with open(checkpoints_path, 'w') as f:
                f.write(model_dir_)

        except Exception as ex:
            # ローカルのコンテナでの実行時には例外となるので、raiseしないでおく
            self._logger.info(ex)

        # コマンド実行
        os.system(cmd)


    def _wait_ai_platform_job_done(
        self,
        project_id,
        job_name,
        output_version
    ):

        """
        AI Platformのジョブが完了するまで待機する

        Args:
            project_id (str): GCPプロジェクトID
            job_name (str): AI Platformのジョブ名
            output_version (int): 出力リソースに付加するバージョン情報

        """

        job_id = f"projects/{project_id}/jobs/{job_name}_{output_version}"
        ml = discovery.build('ml', 'v1', cache_discovery=False)
        request = ml.projects().jobs().get(name=job_id)
        try:
            response = request.execute()
            self._logger.info(f'response: {response}')
            self._logger.info(f'Job status for {job_id}:')
            while response['state'] not in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                self._logger.info(f'    state : {response["state"]}')
                time.sleep(60)
                response = request.execute()
            self._logger.info(f'    state : {response["state"]}')
            self._logger.info(f'    consumedMLUnits : '
                              f'{response["trainingOutput"]["consumedMLUnits"]}')
        except errors.HttpError as err:
            self._logger.info(f'err: {err}')


    def main(self):

        """
        メイン処理
        """

        # Object Detection API の PipelineConfig を作成する
        self._setup_object_detection_pipeline_config(
            self._args.pre_trained_model_name,
            self._args.ai_platform_job_dir_path,
            self._args.num_classes,
            self._args.label_map_path,
            self._args.job_name,
            self._args.tf_record_input_path_train,
            self._args.tf_record_input_path_eval,
            self._args.eval_config_num_examples,
            self._args.train_config_batch_size,
            self._args.quantization
        )

        # プログラムリソースをGCSにコピーする
        self._copy_resource(
            self._args.pre_trained_model_name,
            f'{self._args.ai_platform_job_dir_path}/train/'
            f'{self._args.job_name}_{self._output_version}'
        )

        # AI Platformのジョブを実行する
        self._execute_ai_platform_job(
            self._args.job_name,
            self._args.ai_platform_job_dir_path,
            self._args.pre_trained_model_name,
            self._args.ai_platform_job_region,
            self._args.cloud_yaml_config_path,
            self._args.num_train_steps,
            self._args.config_file_path,
            self._args.checkpoints_path,
            self._args.image_uri,
            self._args.model_dir
        )

        # AI Platformのジョブが完了するまで待機する
        self._wait_ai_platform_job_done(
            self._args.project_id,
            self._args.job_name,
            self._output_version
        )


if __name__ == "__main__":
    print(f"start main : {sys.argv[1:]}")
    try:
        target = ExecuteTrainJobGenerator(sys.argv[1:])
        target.main()
    except Exception as ex:
        print(f"Exception occurred : {ex}")
