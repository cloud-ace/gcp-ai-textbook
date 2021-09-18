import argparse
import datetime
import logging
import os
import sys

import apache_beam as beam
import pandas as pd
import tensorflow
from apache_beam.options.pipeline_options import (
    GoogleCloudOptions, PipelineOptions,
    StandardOptions,
)


class DatalabelingServiceToTfrecordGenerator:


    def __init__(self, args):

        # ロガーの設定
        self._logger = self._get_logger()
        self._logger.info(tensorflow.__version__)

        parser = argparse.ArgumentParser(
            description="launch data labeling service to tfrecord")

        self._logger.info(parser)

        # input params
        # ローカルで実行する「DirectRunner」と、Dataflow上で実行する「DataflowRunner」を選択可能。
        # デフォルトは「DataflowRunner」
        parser.add_argument("--runner", default="DataflowRunner", type=str,
                            help='Execute type local or Dataflow. select in ['
                                 'DirectRunner, DataflowRunner].')
        parser.add_argument("--project_id", default="${your-gcp-project}", type=str,
                            help='GCP project id')
        parser.add_argument("--dataflow_region", default="us-central1", type=str,
                            help='Dataflow job execute region.')

        # Dataflowのジョブスクリプトや中間ファイル格納用のディレクトリを指定
        parser.add_argument("--dataflow_job_dir", type=str,
                            default="gs://${your-gcs-bucket}/dataflow",
                            help='Dataflow job dir in gcs.')

        # データラベリングサービスのアノテーションファイルのパスを指定。これを入力として使用する
        parser.add_argument("--datalabeling_service_annotation_file_path",
                            default="gs://${your-gcs-bucket}/data-labeling-service"
                                    "/export/oid_image_lebel_basketball_labeled_new"
                                    ".json",
                            help='Datalabeling service annotated file path (GCS). Must '
                                 'be json.',
                            type=str)

        # 学習データとバリデーションデータの分割比率を設定。
        # デフォルトは、学習データ8割、バリデーションデータ2割。
        parser.add_argument("--validation_set_rate", type=float, default=0.2,
                            help='Train and validation split rate. This param is '
                                 'validation set rate.')

        # ラベル名のリストを指定
        parser.add_argument("--label_names_list_file_path", type=str,
                            help='Label names list text file path (GCS).',
                            default="gs://${your-gcs-bucket}/data-labeling-service"
                                    "/export/label_names_list.txt")

        # 生成されるtfrecordのディレクトリを指定。
        parser.add_argument("--tfrecord_output_dir", type=str,
                            help='Tfrecord output dir (GCS).',
                            default="gs://${your-gcs-bucket}/tfrecord/image")

        # 生成されるtfrecordのファイルパスを指定。
        parser.add_argument("--file_name_output_gcs_path", type=str,
                            help="Tfrecord output dir(GCS) is stored in this file.")

        # コマンドライン引数をパースする
        self._args = parser.parse_args(args)
        self._logger.info(self._args)

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
            "%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return logger


    def _get_annotations_list(self, json_path, step, validation_set_rate):
        """
        アノテーション情報のリストを１つずつ取り出す

        Args:
            json_path (str): データラベリングサービスのjsonアノテーションデータへのパス
            step (str): "train" or "test"
            validation_set_rate (float): バリデーションデータの比率

        Returns: アノテーション情報の辞書リスト

        """
        # importはBeamのmap関数の中で行う
        import pandas as pd
        import tensorflow as tf

        self._logger.info("get_annotations_list")

        # アノテーションファイルを読み込む
        f = tf.io.gfile.GFile(json_path, 'r')
        df = pd.read_json(f, orient="records", lines=True)

        # ランダムサンプリングする
        df = df.sample(frac=1, random_state=0)

        # 全体数にvalidation_set_rateを掛けて「学習用のサンプル数」を割り出す
        total_record_count = len(df)
        train_set_count = int(total_record_count * (1 - validation_set_rate))

        # train/validation 分割
        if step == "train":
            # 配列のインデックスが先頭から「学習用のサンプル数」まで→学習用
            df = df[0:train_set_count]
        else:
            # 配列のインデックスが「学習用のサンプル数」から最後まで→バリデーション用
            df = df[train_set_count:]

        # 画像のURIとアノテーションを１つずつ辞書リストで返却
        for obj in df.values.tolist():
            yield {"image_uri": obj[1]["image_uri"], "annotations": obj[2]}


    def _create_tf_example(self, element):
        """
    ​
        tf.Exampleを作成する
    ​
        Args:
            element (json): アノテーションデータ（データラベリングサービス）の1レコード
            （image_uriとannotationsのみ）
    ​
        Returns: tf.Example
    ​
        """
        # importはBeamのmap関数の中で行う
        import tensorflow as tf
        from PIL import Image
    ​
        try:
    ​
            if element["image_uri"].split(".")[-1] != "jpg":
                self._logger.info(f"not jpg file: {element['image_uri']}")
                return
    ​
            self._logger.info(f"create_tf_example: {element}")
    ​
            # ①
            f = tf.io.gfile.GFile(element["image_uri"], 'rb')
            # ②
            im = Image.open(f)
    ​
            # 画像(Image)の高さを定義する
            height = im.height
            # 画像の幅を定義する
            width = im.width
            # 画像のファイル名を定義する。もし画像がファイルになければ空文字を入力
            filename = element["image_uri"].encode(
                'utf-8')
            # ③
            img_data = tf.io.gfile.GFile(element["image_uri"], 'rb').read()
    ​
            # バウンディングボックス内ごとの正規化されたX座標の最小値リスト
            # （ボックスごとに1つ）
            xmins = [
                x["annotation_value"]["image_bounding_poly_annotation"][
                    "normalized_bounding_poly"]["normalized_vertices"][
                    0].get("x", 0) for x in
                element[
                    "annotations"]]
    ​
            # バウンディングボックス内ごとの正規化されたX座標の最大値のリスト
            # （ボックスごとに1つ）
            xmaxs = [
                x["annotation_value"]["image_bounding_poly_annotation"][
                    "normalized_bounding_poly"]["normalized_vertices"][
                    1].get("x", 0) for x in
                element[
                    "annotations"]]
    ​
            # バウンディングボックス内ごとの正規化されたY座標の最小値のリスト
            # （ボックスごとに1つ）
            ymins = [
                x["annotation_value"]["image_bounding_poly_annotation"][
                    "normalized_bounding_poly"]["normalized_vertices"][
                    0].get("y", 0) for x in
                element[
                    "annotations"]]
    ​
            # バウンディングボックス内ごとの正規化されたY座標の最大値のリスト
            # （ボックスごとに1つ）
            ymaxs = [
                x["annotation_value"]["image_bounding_poly_annotation"][
                    "normalized_bounding_poly"]["normalized_vertices"][
                    1].get("y", 0) for x in
                element[
                    "annotations"]]
    ​
            # バウンディングボックスのStringクラス名のリスト
            # （ボックスごとに1つ）
            classes_text = [
                x["annotation_value"]["image_bounding_poly_annotation"][
                    "annotation_spec"]["display_name"].encode('utf-8')
                for x in element[
                    "annotations"]]
    ​
            # バウンディングボックスのinteger class idのリスト
            # （ボックスごとに1つ）
            classes = [self._label_names_list.index(
                x["annotation_value"]["image_bounding_poly_annotation"][
                    "annotation_spec"]["display_name"]) for x in
                element[
                    "annotations"]]
    ​
            # ④、⑤
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[height])),
                'image/width': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[width])),
                'image/filename': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[filename])),
                'image/source_id': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[filename])),
                'image/encoded': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img_data])),
                'image/format': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[b'jpg'])),
                'image/object/bbox/xmin': tf.train.Feature(
                    float_list=tf.train.FloatList(value=xmins)),
                'image/object/bbox/xmax': tf.train.Feature(
                    float_list=tf.train.FloatList(value=xmaxs)),
                'image/object/bbox/ymin': tf.train.Feature(
                    float_list=tf.train.FloatList(value=ymins)),
                'image/object/bbox/ymax': tf.train.Feature(
                    float_list=tf.train.FloatList(value=ymaxs)),
                'image/object/class/text': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=classes_text)),
                'image/object/class/label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=classes)),
            }))
            # ⑥
            return tf_example.SerializeToString()
    ​
        except Exception as ex:
            self._logger.info('Exception: %s Element: %s', ex, element)
            raise


    def _get_pipeline_options(self):
        """
        Apache Beamのパイプラインオプションを取得する
    
        Returns: PipelineOptions（パイプラインオプション）
    
        """
        # Apache Beamのパイプラインオプションを設定
        options = PipelineOptions()
        # runnerを選択
        options.view_as(StandardOptions).runner = self._args.runner

        # GCP用のパイプラインオプションを設定
        options.view_as(GoogleCloudOptions).project \
            = self._args.project_id
        options.view_as(GoogleCloudOptions).region \
            = self._args.dataflow_region
        options.view_as(GoogleCloudOptions).staging_location \
            = f"{self._args.dataflow_job_dir}/staging"
        options.view_as(GoogleCloudOptions).temp_location \
            = f"{self._args.dataflow_job_dir}/temp"

        self._logger.info('PipelineOptions: %s', options)
        return options


    def _if_not_null(self, row):
        """
        行がNoneじゃないか判定する（tf.Example生成時に例外が発生したものに関して除外する）

        Args:
            row (dict): 行

        Returns: 行がNoneじゃない場合、True

        """
        return row != None


    def _create_pbtxt_file(self, label_names_list, dest_dir):
        """
        pbtxtファイル（Object Detection APIで使用するラベルマップ情報）を生成する

        Args:
            label_names_list (list): ラベル名のリスト情報
            dest_dir (str): pbtxtファイルの出力先

        """
        with tensorflow.io.gfile.GFile(f"{dest_dir}/label_map.pbtxt", mode='w') as f:
            for index, x in enumerate(label_names_list):
                f.write("item {\n")
                f.write('  name: "{}"\n'.format(x))
                f.write("  id: {}\n".format(index + 1))
                f.write('  display_name: "{}"\n'.format(x))
                f.write("}\n")


    def _set_label_names_list(self):
        """
        ラベル名のリストを取得する
        """
        self._label_names_list = pd.read_csv(self._args.label_names_list_file_path,
                                             header=None, names=["label"])[
            "label"].values.tolist()


    def main(self):
        try:

            # Apache Beamのパイプラインオプションを取得する
            options = self._get_pipeline_options()
            p = beam.Pipeline(options=options)

            # ラベル名のリストを取得する
            self._set_label_names_list()

            # pbtxtファイル（Object Detection APIで使用するラベルマップ情報）を生成する
            self._create_pbtxt_file(self._label_names_list,
                                    f"{self._args.tfrecord_output_dir}"
                                    f"/{self._output_version}")

            # パイプライン処理（train`と`validation`でそれぞれ生成）
            for step in ['train', 'validation']:
                _ = (p | f"json_path_{step}" >>
                     beam.Create([self._args.datalabeling_service_annotation_file_path])
                     # データラベリングサービスのアノテーション情報を読み込む
                     | f"get_annotations_list_{step}" >>
                     beam.FlatMap(self._get_annotations_list, step,
                                  self._args.validation_set_rate)
                     # tf.Exampleを作成する
                     | f"create_tf_example_{step}" >>
                     beam.Map(self._create_tf_example)
                     # 行がNoneじゃないか判定する
                     # （tf.Example生成時に例外が発生したものに関して除外する）
                     | f"filter_{step}" >>
                     beam.Filter(self._if_not_null)
                     # TFRecordを作成する
                     # （上記で作成したtf.Exampleをひとまとまりにしてファイルにする）
                     | f"write_{step}" >>
                     beam.io.tfrecordio.WriteToTFRecord(
                         self._args.tfrecord_output_dir +
                         f"/{self._output_version}/{step}/data",
                         file_name_suffix=".tfrecord")
                     )

            # ジョブが完了するまで待機する
            p.run().wait_until_finish()

            self._logger.info(f"[job done] : datalabeling_service_to_tfrecord_generator")

            # Create directory for storing output path
            os.makedirs(os.path.dirname(self._args.file_name_output_gcs_path))

            with open(self._args.file_name_output_gcs_path, 'w') as f:
                f.write(f"{self._args.tfrecord_output_dir}/{self._output_version}")

        except Exception as ex:
            self._logger.info(f"Exception: {ex}")


if __name__ == "__main__":
    print(f"start main : {sys.argv[1:]}")
    try:
        target = DatalabelingServiceToTfrecordGenerator(sys.argv[1:])
        target.main()
    except Exception as ex:
        print(f"Exception occurred : {ex}")
