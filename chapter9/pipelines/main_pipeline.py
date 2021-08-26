import logging
import pathlib

import kfp
# loggingの設定
from builtins import str

import kfp
from kfp import dsl

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# componentの設定ファイルまでのパス
file_path = pathlib.Path(__file__).resolve().parents[1]
component_root_url = file_path.joinpath("components/")

# config fileを読み込む
tfrecord_genereator_config = str(
    component_root_url.joinpath('datalabeling_service_to_tfrecord_generator/component.yaml'))
trainer_config = str(component_root_url.joinpath('execute_train_job_generator/component.yaml'))
model_compiler_config = str(component_root_url.joinpath('model_compile_generator/component.yaml'))

# Component作成
tfrecord_generator = kfp.components.load_component_from_file(tfrecord_genereator_config)
training_job_generator = kfp.components.load_component_from_file(trainer_config)
model_compile_generator = kfp.components.load_component_from_file(model_compiler_config)


# Pipeline
@dsl.pipeline(
    name="Train model and compile it from tfrecord.",
    description="tfrecordからObject Detection Modelを訓練し、Saved Model形式で保存する"
)
def pipeline(
        # common parameter
        project_id="${project_id}",
        region="us-central1",
        # for tfrecord_generator
        apache_beam_runner="DataflowRunner",
        dataflow_job_directory="${GCS path}",
        datalabeling_service_annotation_file_path="gs://shoseki-ai-platform/data-labeling-service/export/oid_image_lebel_basketball_labeled_new.json",
        validation_set_rate=0.2,
        label_names_list_file_path="gs://shoseki-ai-platform/data-labeling-service/export/label_names_list.txt",
        tfrecord_output_dir="${GCS path}",
        # training_job_generator
        ai_platform_job_directory="${GCS path}",
        ai_platform_job_name="${job name}",
        number_of_classes=2,
        batch_size=12,
        training_steps=100,
        number_of_samples_in_eval=100,
        ai_platform_model_dir_name="",
        image_uri_for_training="gcr.io/ai-shoseki-ai-platform/execute_train_job_generator:latest-1.15.2-gpu-py3",
        # model_compile_generator
        model_export_path="${GCS path}",
        model_type="edge_tpu_tflite"
):
    tfrecord_generator_op = tfrecord_generator(
        project_id=project_id,
        apache_beam_runner=apache_beam_runner,
        dataflow_region=region,
        dataflow_job_directory=dataflow_job_directory,
        datalabeling_service_annotation_file_path=datalabeling_service_annotation_file_path,
        validation_set_rate=validation_set_rate,
        label_names_list_file_path=label_names_list_file_path,
        tfrecord_output_dir=tfrecord_output_dir,
    ).set_image_pull_policy('Always')
    training_job_generator_op = training_job_generator(
        project_id=project_id,
        ai_platform_region=region,
        ai_platform_job_directory=ai_platform_job_directory,
        ai_platform_job_name=ai_platform_job_name,
        ai_platform_model_dir_name=ai_platform_model_dir_name,
        number_of_classes=number_of_classes,
        label_map_path=f"{tfrecord_generator_op.outputs['output_gcs_path']}/label_map.pbtxt",
        tfrecord_input_path_for_train=f"{tfrecord_generator_op.outputs['output_gcs_path']}/train/data-00000-of-00001.tfrecord",
        tfrecord_input_path_for_eval=f"{tfrecord_generator_op.outputs['output_gcs_path']}/validation/data-00000-of-00001.tfrecord",
        number_of_samples_in_eval=number_of_samples_in_eval,
        batch_size=batch_size,
        training_steps=training_steps,
        config_file_path="/pipelines/component/src/models/research/cloud.yaml",
        pretrained_model="ssd_mobilenet_v2_oid_v4_2018_12_12",
        quantization=False,
        image_uri=image_uri_for_training
    ).set_image_pull_policy('Always')
    checkpoint_getter = dsl.ContainerOp(
        name="checkpoints_getter",
        image="gcr.io/cloud-builders/gsutil:latest",
        command=["sh", "-c"],

        arguments=[
            f"gsutil ls {training_job_generator_op.outputs['checkpoints_directory_path']} | sed -r 's/.*model\.ckpt-([0-9]*)\..*/\\1/g' | sort -n -r | head -n 1 > output.txt"
        ],
        file_outputs={
            "checkpoints_num": "output.txt"
        }
    ).set_image_pull_policy('Always')
    model_compile_generator_op = model_compile_generator(
        project_id=project_id,
        checkpoint_path=f"{training_job_generator_op.outputs['checkpoints_directory_path']}/model.ckpt-{checkpoint_getter.outputs['checkpoints_num']}",
        config_path=training_job_generator_op.outputs['config_file_path'],
        model_export_path=model_export_path,
        model_type=model_type
    ).set_image_pull_policy('Always')


if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(pipeline, __file__ + ".tar.gz")
