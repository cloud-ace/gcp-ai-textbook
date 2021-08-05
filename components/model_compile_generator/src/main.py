import argparse
import datetime
import glob
import logging
import os
import sys
from http import HTTPStatus

import tensorflow as tf
from flask import Response


class ModelCompileGenerator:


    def __init__(self, args):

        # ロガーの設定
        self._logger = self._get_logger()
        self._logger.info(f"TensorFlow version : {tf.__version__}")

        # 定数
        self.MODEL_NAME = "oid_ssd_mobilenet_v2"
        self.WORK_PATH = f"exported_graphs"
        self.TF_LITE_MODEL_FILE_PATH = f"{self.MODEL_NAME}_quantized.tflite"
        self.EDGE_TPU_MODEL_FILE_PATH = f"{self.MODEL_NAME}_quantized_edgetpu.tflite"
        self.ODA_ROOT_PATH = "/pipelines/component/src/models/research/"

        parser = argparse.ArgumentParser(
            description="launch dataflow template")

        parser.add_argument("--project_id", type=str, default="ai-shoseki-ai-platform", help='GCP project id')
        parser.add_argument("--trained_checkpoint_prefix", type=str,
                            default="gs://shoseki-ai-platform/custom_model_new/train/object_detection_train/model_dir/model.ckpt"
                                    "-10002",
                            help='AI Platform trained model checkpoint file path prefix (GCS). ')
        parser.add_argument("--pipeline_config_path", type=str,
                            default="gs://shoseki-ai-platform/custom_model_new/train/object_detection_train"
                                    "/ssd_mobilenet_v2_oid_v4_2018_12_12/pipeline_new.config",
                            help='Object Detection API pipeline config file path (GCS).')
        parser.add_argument("--model_export_dir", type=str,
                            default="gs://shoseki-ai-platform/custom_model_new/train/object_detection_train/model_dir/export",
                            help='for edgetpu compiled tflite file export path (GCS) .')
        parser.add_argument("--model_type", type=str,
                            default="saved_model",
                            help='Select the model type in ["saved_model", "edge_tpu_tflite"]')

        # コマンドライン引数をパースする
        self._args = parser.parse_args(args)
        self._logger.info(f'args: {self._args}')

        # 出力リソースに付加するバージョン情報
        self._output_version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


    def _get_logger(self) -> object:
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


    def _export_tflite_ssd_graph(self, pipeline_config_path: str, trained_checkpoint_prefix: str) -> object:
        """
        tflite_ssd_graph（tfliteと互換性があるfrozenグラフ）を生成する

        Args:
            pipeline_config_path (str): pipeline_configへのパス
            trained_checkpoint_prefix (str): チェックポイントファイルへのパス

        """
        cmd = f"""python {os.path.join(self.ODA_ROOT_PATH, "object_detection/export_tflite_ssd_graph.py")}  \
                      --pipeline_config_path {pipeline_config_path} \
                      --trained_checkpoint_prefix  {trained_checkpoint_prefix} \
                      --output_directory {os.path.join(self.ODA_ROOT_PATH, self.WORK_PATH)}  \
                      --add_postprocessing_op True
                """
        self._logger.info(f"cmd: {cmd}")
        self._logger.info(f"result: {os.system(cmd)}")
        files = glob.glob(os.path.join(self.ODA_ROOT_PATH, self.WORK_PATH, '*'))
        if len(files) == 0:
            raise Exception("Saved Model for tflite is not exported.")


    def _export_frozen_inference_graph(self, pipeline_config_path: str, trained_checkpoint_prefix: str) -> object:
        """
        frozen_inference_graph（パラメータが凍結（固定化）された推論用のグラフ）を生成する

        Args:
            pipeline_config_path (str): pipeline_configへのパス
            trained_checkpoint_prefix (str): チェックポイントファイルへのパス

        """

        cmd = f"""python {os.path.join(self.ODA_ROOT_PATH, "object_detection/export_inference_graph.py")} \
                      --input_type encoded_image_string_tensor \
                      --input_shape 1,300,300,3 \
                      --pipeline_config_path {pipeline_config_path} \
                      --trained_checkpoint_prefix  {trained_checkpoint_prefix} \
                      --output_directory {os.path.join(self.ODA_ROOT_PATH, self.WORK_PATH)}  \
        """
        self._logger.info(f"cmd: {cmd}")
        self._logger.info(f"result: {os.system(cmd)}")
        files = glob.glob(os.path.join(self.ODA_ROOT_PATH, self.WORK_PATH, "*"))
        if len(files) == 0:
            raise Exception("Saved Model is not exported.")
        self._logger.info(f"Exported inference graph is {files}")


    def _convert_to_tflite(self, tf_lite_model_file_path: str) -> object:
        """
        frozenグラフをtfliteモデルに変換

        Args:
            tf_lite_model_file_path (str): tfliteモデルが生成されるパス

        """
        cmd = f"""tflite_convert \
                        --graph_def_file {os.path.join(self.ODA_ROOT_PATH, self.WORK_PATH, "tflite_graph.pb")} \
                        --output_file {os.path.join(self.ODA_ROOT_PATH, self.WORK_PATH, tf_lite_model_file_path)} \
                        --input_arrays=normalized_input_image_tensor \
                        --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
                        --input_shape=1,300,300,3 \
                        --allow_custom_ops \
                        --inference_input_type=QUANTIZED_UINT8 \
                        --mean_values=128 \
                        --std_dev_values=127 \
                        --default_ranges_min=0 \
                        --default_ranges_max=255 \
                        --inference_type=QUANTIZED_UINT8
        """
        self._logger.info(f"cmd: {cmd}")
        self._logger.info(f"result: {os.system(cmd)}")
        files = glob.glob(os.path.join(self.ODA_ROOT_PATH, self.WORK_PATH, tf_lite_model_file_path))
        if len(files) == 0:
            raise Exception("tflite Model is not exported.")


    def _compile_for_edgetpu(self, tf_lite_model_file_path: str, edge_tpu_model_file_path: str) -> object:
        """
        EdgeTPU用にモデルをコンパイルする

        Args:
            tf_lite_model_file_path (str): tfliteモデルのパス

        """
        cmd = f"""edgetpu_compiler -s \
                    {os.path.join(self.ODA_ROOT_PATH, self.WORK_PATH, tf_lite_model_file_path)} \
                    -o {os.path.join(self.ODA_ROOT_PATH, self.WORK_PATH)}"""
        self._logger.info(f"cmd: {cmd}")
        self._logger.info(f"result: {os.system(cmd)}")
        files = glob.glob(os.path.join(self.ODA_ROOT_PATH, self.WORK_PATH, edge_tpu_model_file_path))
        if len(files) == 0:
            raise Exception("EdgeTPU tflite Model is not exported.")


    def _export_to_gcs(self, model_export_dir: str) -> object:
        """
        GCSへモデルファイルをエクスポートする

        Args:
            model_export_dir (str): コピー先ディレクトリ

        """
        files = [p for p in
                 glob.glob(os.path.join(self.ODA_ROOT_PATH, self.WORK_PATH, '**'), recursive=True)
                 if os.path.isfile(p)]

        # 移送先のディレクトリを生成
        dest_dir = os.path.join(model_export_dir, self._output_version)
        tf.io.gfile.makedirs(dest_dir)

        for file_path in files:
            # コピー実行
            dest_path = os.path.join(dest_dir, os.path.basename(file_path))
            tf.io.gfile.copy(file_path, dest_path, overwrite=True)
            while not tf.io.gfile.exists(dest_path):
                os.sleep(1)
            self._logger.info(f"Model export done. Destination is {dest_path}")


    def main(self):

        if self._args.model_type == "saved_model":
            # saved model（通常 or クラウド用のモデル）を生成するフロー

            # チェックポイントファイルから、frozen_inference_graph（パラメータが凍結（固定化）された推論用のグラフ）を生成する
            self._export_frozen_inference_graph(self._args.pipeline_config_path, self._args.trained_checkpoint_prefix)

        elif self._args.model_type == "edge_tpu_tflite":
            # tflite model（Edge TPU用のモデル）を生成するフロー

            # チェックポイントファイルから、tflite_ssd_graph（tfliteと互換性があるfrozenグラフ）を生成する
            self._export_tflite_ssd_graph(self._args.pipeline_config_path, self._args.trained_checkpoint_prefix)
            # frozenグラフをtfliteモデルに変換
            self._convert_to_tflite(self.TF_LITE_MODEL_FILE_PATH)
            # tfliteモデルをEdgeTPU用にコンパイル
            self._compile_for_edgetpu(self.TF_LITE_MODEL_FILE_PATH, self.EDGE_TPU_MODEL_FILE_PATH)

        # 生成したモデルをGCSにコピーする
        self._export_to_gcs(self._args.model_export_dir)

        return Response(status=HTTPStatus.OK)


if __name__ == "__main__":
    print(f"start main : {sys.argv[1:]}")
    try:
        target = ModelCompileGenerator(sys.argv[1:])
        target.main()
    except Exception as ex:
        print(f"Exception occurred : {ex}")
