import os
import sys
from http import HTTPStatus
from pathlib import Path

import pytest
from flask import Response


sys.path.append(Path(os.path.dirname(__file__)).joinpath(
    "../../model_compile_generator/src").resolve().as_posix())
sys.path.append(Path(os.path.dirname(__file__)).joinpath(
    "../..").resolve().as_posix())

from model_compile_generator.src.main import ModelCompileGenerator


class TestModelCompileGenerator:
    _project_id = os.getenv("PROJECT_ID")


    @pytest.mark.parametrize("trained_checkpoint_prefix, pipeline_config_path, model_export_dir, model_type, expected", [
        ("model.ckpt-120007",
         "pipeline.config",
         "output",
         "edge_tpu_tflite", Response(status=HTTPStatus.OK)),
        ("model.ckpt-120007",
         "pipeline.config",
         "output",
         "saved_model", Response(status=HTTPStatus.OK))
    ]
                             )
    def test_method_get_annotations_list(self,
        trained_checkpoint_prefix,
        pipeline_config_path,
        model_export_dir,
        model_type,
        expected
    ):
        params = ['--trained_checkpoint_prefix',
                  os.path.join(os.path.dirname(__file__), f"test_data/{model_type}/input/{trained_checkpoint_prefix}"),
                  '--pipeline_config_path',
                  os.path.join(os.path.dirname(__file__), f"test_data/{model_type}/input/{pipeline_config_path}"),
                  '--model_export_dir', os.path.join(os.path.dirname(__file__), f"test_data/{model_type}/{model_export_dir}"),
                  '--model_type', model_type]

        target = ModelCompileGenerator(params)
        result = target.main()
        assert result.status == expected.status
