import setuptools
import os
from setuptools import setup

version = "0.5.12"
maca_ai_version = os.environ.get('MACA_AI_VERSION', '0.0.0.0')
version += "+maca"+maca_ai_version
setup(
    version=version
)