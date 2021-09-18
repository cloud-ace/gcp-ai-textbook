# loggigの設定
from kfp import dsl


@dsl.pipeline(
    name="Get the newest checkpoint",
    description=""
)
def pipeline(
        checkpoints_directory_path="gs://${your-gcs-bucket}/train/export"
):
    checkpoint_getter = dsl.ContainerOp(
        name="checkpoints_getter",
        image="gcr.io/cloud-builders/gsutil:latest",
        command=["sh","-c"],

        arguments=[
            f"gsutil ls {checkpoints_directory_path} | sed -r 's/.*model\.ckpt-([0-9]*)\..*/\\1/g' | sort -n -r | head -n 1 > output.txt"
        ],
        file_outputs={
            "checkpoints_num": "output.txt"
        }
    ).set_image_pull_policy('Always')


if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(pipeline, __file__ + ".tar.gz")
