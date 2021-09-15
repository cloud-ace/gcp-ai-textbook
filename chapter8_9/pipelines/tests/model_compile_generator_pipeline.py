import logging
import pathlib
# loggigの設定
from builtins import str

import kfp
from kfp import dsl

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# componentの設定ファイルまでのパス
file_path = pathlib.Path(__file__).resolve().parents[2]
component_root_url = file_path.joinpath("components/")
config_url = str(component_root_url.joinpath('model_compile_generator/component.yaml'))
edgetpu_generator = kfp.components.load_component_from_file(config_url)
logger.info("component file path:{}".format(config_url))


# pipeline
@dsl.pipeline(
    name="edge tpu compile tests",
    description=""
)
def pipeline(
        project_id="${your-gcp-project}",
        checkpoint_path="gs://${your-gcs-bucket}/custom_model_new/train/object_detection_train/model_dir/model.ckpt-2000",
        config_path="gs://${your-gcs-bucket}/custom_model_new/train/object_detection_train/ssd_mobilenet_v2_oid_v4_2018_12_12/pipeline_new.config",
        model_export_path="gs://${your-gcs-bucket}/custom_model_new/train/object_detection_train/model_dir/export/Servo/1587553106",
        model_type="edge_tpu_tflite"
):
     edgetpu_generator_op = edgetpu_generator(
         project_id=project_id,
         checkpoint_path=checkpoint_path,
         config_path=config_path,
         model_export_path=model_export_path,
         model_type=model_type
    ).set_image_pull_policy('Always')


if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(pipeline, __file__ + ".tar.gz")