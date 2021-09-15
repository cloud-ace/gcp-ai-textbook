import os
import sys
from pathlib import Path

import pytest
import tensorflow as tf


sys.path.append(Path(os.path.dirname(__file__)).joinpath(
    "../../datalabeling_service_to_tfrecord_generator/src").resolve().as_posix())

from main import DatalabelingServiceToTfrecordGenerator


class TestDatalabelingServiceToTfrecordGenerator:

    _project_id = os.getenv("PROJECT_ID")


    @pytest.mark.parametrize("json_path, step, validation_set_rate, expected", [
        (os.path.join(
            os.path.dirname(__file__), "test_data/test_datalabeling_service_annotation_data.json"), "train", 0.2, 80),
        (os.path.join(
            os.path.dirname(__file__), "test_data/test_datalabeling_service_annotation_data.json"), "test", 0.33, 33),
    ]
                             )
    def test_method_get_annotations_list(self, json_path, step, validation_set_rate, expected):
        params = ['--runner', 'DirectRunner']
        target = DatalabelingServiceToTfrecordGenerator(params)
        assert expected == len(list(target._get_annotations_list(json_path, step, validation_set_rate)))


    @pytest.mark.parametrize("element, expected", [
        ({"image_uri": os.path.join(os.path.dirname(__file__), "test_data/1064px-TMac_over_Deshawn.jpg"),
          "annotations": [
              {
                  "name": "projects\/your-project-id\/datasets\/your-dataset-id\/annotatedDatasets\/9647721955250294567"
                          "-9647721955250292517\/examples\/5824192384389503608\/annotations\/9779710009781250562",
                  "annotation_source": 3,
                  "annotation_value": {
                      "image_bounding_poly_annotation": {
                          "annotation_spec": {
                              "display_name": "Basketball Shooter"
                          },
                          "normalized_bounding_poly": {
                              "normalized_vertices": [
                                  {
                                      "x": 0.14562967,
                                      "y": 0.15411143
                                  },
                                  {
                                      "x": 0.82441801,
                                      "y": 1
                                  }
                              ]
                          }
                      }
                  },
                  "annotation_metadata": {
                      "operator_metadata": {

                      }
                  }
              }
          ]},
         tf.train.Example(features=tf.train.Features(feature={
             'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[1200])),
             'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[1064])),
             'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(
                 value=[os.path.join(os.path.dirname(__file__), "test_data/1064px-TMac_over_Deshawn.jpg").encode('utf-8')])),
             'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(
                 value=[os.path.join(os.path.dirname(__file__), "test_data/1064px-TMac_over_Deshawn.jpg").encode('utf-8')])),
             'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                 tf.io.gfile.GFile(os.path.join(os.path.dirname(__file__), "test_data/1064px-TMac_over_Deshawn.jpg"),
                                   'rb').read()])),
             'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),
             'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=[0.1456296741962433])),
             'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=[0.8244180083274841])),
             'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=[0.15411143004894257])),
             'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])),
             'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"Basketball Shooter"])),
             'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
         }))
         )
    ])
    def test_method_create_tf_example(self, element, expected):
        params = ['--runner', 'DirectRunner', '--label_names_list_file_path', os.path.join(
            os.path.dirname(__file__), "test_data/label_names_list.txt")]
        target = DatalabelingServiceToTfrecordGenerator(params)
        target._set_label_names_list()
        result = tf.train.Example.FromString(target._create_tf_example(element))
        assert expected == result
