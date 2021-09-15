from kfp import dsl
from kfp.components import load_component_from_url

# componentの定義ファイルのURI
COMPONENT_SPEC_URI = 'https://storage.googleapis.com/aihub-content-test-data/ai-hub-assets/images_to_od_tfrecords/component.yaml'
image_to_tfrecords_op = load_component_from_url(COMPONENT_SPEC_URI)

# parameter
COCO_BUCKET = 'gs://images.cocodataset.org'
TEST_ANNOTATIONS = 'gs://aihub-content-test-data/images_to_od_tfrecords/annotations'
PROJECT_ID = "ca-highbird-test"
GCS_OUTPUT_DIR = f"gs://{PROJECT_ID}/kubfeflow_pipelines"

# Specify pipeline argument values
arguments = {
    'train_image_dir': f'{COCO_BUCKET}/train2017',
    'test_image_dir': f'{COCO_BUCKET}/test2017',
    'val_image_dir': f'{COCO_BUCKET}/val2017',
    'train_annotations_file': f'{TEST_ANNOTATIONS}/instances_train2017-10000.json',
    'test_annotations_file': f'{TEST_ANNOTATIONS}/image_info_test-dev2017-4000.json',
    'val_annotations_file': f'{TEST_ANNOTATIONS}/instances_val2017-1000.json',
    'output': GCS_OUTPUT_DIR,
    'max_train_images': 100,
    'max_test_images': 40,
    'max_val_images': 10,
    'require_annotations': True,
    'project': PROJECT_ID,
    'runner': 'dataflow',
}


@dsl.pipeline(
    name="coco test pipeline",
    description='''https://aihub.cloud.google.com/p/products%2Ff99a27fc-e093-459d-8a12-6333f78c073c に
    書いてある内容をテストする。'''
)
def pipeline(
        train_image_dir=arguments['train_image_dir'],
        test_image_dir=arguments['test_image_dir'],
        val_image_dir=arguments['val_image_dir'],
        train_annotations_file=arguments['train_annotations_file'],
        test_annotations_file=arguments['test_annotations_file'],
        val_annotations_file=arguments['val_annotations_file'],
        output=arguments['output'],
        max_train_images=arguments['max_train_images'],
        max_test_images=arguments['max_test_images'],
        max_val_images=arguments['max_val_images'],
        require_annotations=arguments['require_annotations'],
        shard_size=50,
        project=arguments['project'],
        runner=arguments['runner'],
        num_workers=2
):
    image_to_tfrecords_op(
        train_image_dir,
        test_image_dir,
        val_image_dir,
        train_annotations_file,
        test_annotations_file,
        val_annotations_file,
        output,
        max_train_images,
        max_test_images,
        max_val_images,
        require_annotations,
        shard_size,
        project,
        runner,
        num_workers
    )  # .apply(use_gcp_secret('user-gcp-sa')) # AI Platform Pipelinesでは知らせるときの


if __name__ == '__main__':
    pipeline_func = pipeline
    pipeline_filename = pipeline_func.__name__ + '.tar.gz'
    from kfp.compiler import Compiler

    Compiler().compile(pipeline_func, pipeline_filename)
