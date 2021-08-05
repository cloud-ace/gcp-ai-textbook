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
config_url = str(component_root_url.joinpath('execute_train_job_generator/component.yaml'))
training_job_generator = kfp.components.load_component_from_file(config_url)
logger.info("component file path:{}".format(config_url))


# pipeline
@dsl.pipeline(
    name="datalabeling service tests",
    description=""
)
def pipeline(
        project_id="ai-shoseki-ai-platform",
        config_file_path="object_detection/samples/cloud/cloud.yml",
        ai_platform_region="us-central1",
        ai_platform_job_directory="gs://shoseki-ai-platform",
        number_of_classes=2,
        label_map_path="gs://shoseki-ai-platform/tfrecord/image/20200411152510/train/label_map.pbtxt",
        tfrecord_input_path_for_train="gs://shoseki-ai-platform/tfrecord/image/20200411152510/validation/data-00000-of-00001.tfrecord",
        tfrecord_input_path_for_eval="gs://shoseki-ai-platform/tfrecord/image/20200411152510/validation/data-00000-of-00001.tfrecord",
        batch_size=12,
        training_steps=100,
        number_of_samples_in_eval=100,
):
    training_job_generator_op = training_job_generator(
        project_id=project_id,
        config_file_path=config_file_path,
        ai_platform_region=ai_platform_region,
        ai_platform_job_directory=ai_platform_job_directory,
        number_of_classes=number_of_classes,
        label_map_path=label_map_path,
        tfrecord_input_path_for_train=tfrecord_input_path_for_train,
        tfrecord_input_path_for_eval=tfrecord_input_path_for_eval,
        batch_size=batch_size,
        training_steps=training_steps,
        number_of_samples_in_eval=number_of_samples_in_eval
    ).set_image_pull_policy('Always')


if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(pipeline, __file__ + ".tar.gz")
