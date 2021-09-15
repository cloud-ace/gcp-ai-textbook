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
config_url = str(component_root_url.joinpath('datalabeling_service_to_tfrecord_generator/component.yaml'))
tfrecord_generator = kfp.components.load_component_from_file(config_url)
logger.info("component file path:{}".format(config_url))


# pipeline
@dsl.pipeline(
    name="datalabeling service tests",
    description=""
)
def pipeline(
        project_id="${your-gcp-project}",
        apache_beam_runner="DataflowRunner",
        dataflow_region="us-central1",
        dataflow_job_directory="gs://${your-gcs-bucket}",
        datalabeling_service_annotation_file_path="gs://${your-gcs-bucket}/data-labeling-service/export/oid_image_lebel_basketball_labeled.json",
        validation_set_rate=0.2,
        label_names_list_file_path="gs://${your-gcs-bucket}/data-labeling-service/export/label_names_list.txt",
        tfrecord_output_dir="gs://${your-gcs-bucket}/tfrecord-test/image"
):
    tfrecord_generator_op = tfrecord_generator(
        project_id=project_id,
        apache_beam_runner=apache_beam_runner,
        dataflow_region=dataflow_region,
        dataflow_job_directory=dataflow_job_directory,
        datalabeling_service_annotation_file_path=datalabeling_service_annotation_file_path,
        validation_set_rate=validation_set_rate,
        label_names_list_file_path=label_names_list_file_path,
        tfrecord_output_dir=tfrecord_output_dir
    ).set_image_pull_policy('Always')


if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(pipeline, __file__ + ".tar.gz")
