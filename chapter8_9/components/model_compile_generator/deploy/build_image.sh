#!/bin/bash -e
image_name=gcr.io/${your-gcp-project}/model-compile-generator
image_tag=v1
full_image_name=${image_name}:${image_tag}
base_image_tag=3.7-slim

cd "$(dirname "$0")"
docker build --build-arg BASE_IMAGE_TAG=${base_image_tag} -t "${full_image_name}" . # buil-argはDockerfileで定義済み
docker push "$full_image_name"

# Output the strict image name (which contains the sha256 image digest)
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"