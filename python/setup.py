# import setuptools
# import os
# from setuptools import setup

# version = "0.5.14"
# maca_ai_version = os.environ.get('MACA_AI_VERSION', '0.0.0.0')
# version += "+maca"+maca_ai_version
# setup(
#     version=version
# )

"""sglang build hooks.

Rust extensions are auto-discovered from the cargo workspace in ../rust: every
crate whose Cargo.toml declares

    [package.metadata.sglang]
    python-module = "sglang.srt.<pkg>._core"   # import path inside the wheel
    debug = false                              # optional RustExtension knob
    features = ["python"]                      # optional cargo features to enable

is built as a PyO3 extension module at that import path. Adding a new extension
crate therefore needs no pyproject changes — declare the metadata in the crate.

Two filters can narrow the discovered set:

- [tool.sglang] rust-extensions in the active pyproject.toml: a list of
  case-insensitive substrings of the target module. Platform pyprojects use
  this to build a subset (e.g. pyproject_other.toml builds only "multimodal";
  grpc needs proto/tonic and is intentionally CUDA-only).
- SGLANG_BUILD_RUST_EXTS env var, applied at build time on top of the above:
  unset or "all" builds everything, "none" builds nothing, and a
  comma-separated list matches substrings, e.g. "grpc" matches
  "sglang.srt.grpc._core". It is read directly from os.environ instead of
  sglang.srt.environ, which is not importable until the package is built.
"""

import json
import os
from setuptools import setup

version = "0.5.18"
maca_ai_version = os.environ.get('MACA_AI_VERSION', '0.0.0.0')
version += "+maca"+maca_ai_version
setup(
    version=version
)