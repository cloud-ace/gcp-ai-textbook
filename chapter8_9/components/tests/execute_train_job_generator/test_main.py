import os
import sys
from pathlib import Path

import pytest


sys.path.append(Path(os.path.dirname(__file__)).joinpath(
    "../..").resolve().as_posix())

from execute_train_job_generator.src.main import ExecuteTrainJobGenerator


class TestExecuteTrainJobGenerator:
    _project_id = os.getenv("PROJECT_ID")


    @pytest.mark.parametrize("pre_trained_model_name,"
                             "ai_platform_job_dir_path,"
                             "num_classes,"
                             "label_map_path,"
                             "job_name,"
                             "tf_record_input_path_train,"
                             "tf_record_input_path_eval,"
                             "eval_config_num_examples,"
                             "train_config_batch_size,"
                             "quantization,"
                             "expected", [
                                 ("ssd_mobilenet_v2_oid_v4_2018_12_12",
                                  "gs://your-bucket/custom_model_new",
                                  2,
                                  "label_map.pbtxt",
                                  "object_detection_train",
                                  "gs://your-bucket/tfrecord/image/20200411152510/train/data-00000-of-00001.tfrecord",
                                  "gs://your-bucket/tfrecord/image/20200411152510/validation/data-00000-of-00001"
                                  ".tfrecord",
                                  100,
                                  12,
                                  False,
                                  ""
                                  )
                             ]
                             )
    def test_setup_object_detection_pipeline_config(
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
        quantization,
        expected
    ):
        params = ['--label_map_path', os.path.join(os.path.dirname(__file__), f"test_data/input/{label_map_path}")]
        target = ExecuteTrainJobGenerator(params)
        target.main()
