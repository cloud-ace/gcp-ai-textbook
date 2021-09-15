"""Setup script for object_detection with TF1.0."""
import os

from setuptools import find_packages
from setuptools import setup


# AI Platform Train 実行時に追加でインストールしたいライブラリはここで指定する
# https://cloud.google.com/ai-platform/training/docs/packaging-trainer?hl=ja#adding_standard_pypi_dependencies
REQUIRED_PACKAGES = ['pillow', 'lxml', 'matplotlib', 'Cython',
                     'contextlib2', 'tf-slim', 'pycocotools', 'lvis',
                     'scipy', 'pandas', 'pip==21.0.1', 'google-resumable-media<0.5.0dev,>=0.3.1',
                     'six==1.15.0', 'google-api-python-client==1.12.8',
                     'numpy==1.19.5', 'python-dateutil>=2.8.0',
                     'Flask==1.1.2',
                     'pytest==6.2.2',
                     'opencv-python-headless==4.2.0.34']

setup(
    name='object_detection',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=(
        [p for p in find_packages() if p.startswith('object_detection')] +
        find_packages(where=os.path.join('.', 'slim'))),
    package_dir={
        'datasets': os.path.join('slim', 'datasets'),
        'nets': os.path.join('slim', 'nets'),
        'preprocessing': os.path.join('slim', 'preprocessing'),
        'deployment': os.path.join('slim', 'deployment'),
        'scripts': os.path.join('slim', 'scripts'),
    },
    description='Tensorflow Object Detection Library with TF1.0',
    python_requires='>3.6',
)
