#!/bin/bash -e
image_name=gcr.io/ai-shoseki-ai-platform/datalabeling-service-to-tfrecordgenerator # Specify the image name here
image_tag=v5
full_image_name=${image_name}:${image_tag}
base_image_tag=3.8-slim

cd "$(dirname "$0")"
docker build --build-arg BASE_IMAGE_TAG=${base_image_tag} -t "${full_image_name}" . # buil-argはDockerfileで定義済み
docker push "$full_image_name"

# Output the strict image name (which contains the sha256 image digest)
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"