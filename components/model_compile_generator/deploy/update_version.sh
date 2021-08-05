#!/bin/bash
VERSION=v1
which gsed >/dev/null 2>&1
if [ $? -eq 0 ]; then
        echo "alias sed to gsed." >&2
        shopt -s expand_aliases
        alias sed=gsed
fi

sed -i'' "s/\(^image_tag=\).*$/\1${VERSION}/g" build_image.sh
sed -i'' "s/\(gcr.io[^:]*\):.*/\1:${VERSION}/g" component.yaml