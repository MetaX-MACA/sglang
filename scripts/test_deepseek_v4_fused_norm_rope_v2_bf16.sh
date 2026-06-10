#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/python:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

CUCC_BIN="${CUCC_BIN:-}"
if [[ -z "${CUCC_BIN}" ]]; then
  for candidate in \
    "${CUCC_PATH:-}/bin/cucc" \
    "${MACA_PATH:-}/tools/cu-bridge/bin/cucc" \
    /opt/maca/tools/cu-bridge/bin/cucc \
    /opt/maca-3.5.3/tools/cu-bridge/bin/cucc; do
    if [[ -x "${candidate}" ]]; then
      CUCC_BIN="${candidate}"
      break
    fi
  done
fi

if ! command -v nvcc >/dev/null 2>&1 && [[ -n "${CUCC_BIN}" ]]; then
  CUDA_SHIM_DIR="${ROOT_DIR}/.cache/cucc-nvcc"
  mkdir -p "${CUDA_SHIM_DIR}/bin"
  REAL_CUDA_PATH="$(cd "$(dirname "${CUCC_BIN}")/.." && pwd)"
  cat > "${CUDA_SHIM_DIR}/bin/nvcc" <<EOF
#!/usr/bin/env bash
export CUCC_PATH="${REAL_CUDA_PATH}"
export CUDA_PATH="${REAL_CUDA_PATH}"
exec "${CUCC_BIN}" "\$@"
EOF
  chmod +x "${CUDA_SHIM_DIR}/bin/nvcc"
  ln -sfn "${REAL_CUDA_PATH}/include" "${CUDA_SHIM_DIR}/include"
  MACA_LIB_DIR="${MACA_PATH:-/opt/maca}/lib64"
  [[ -d "${MACA_LIB_DIR}" ]] || MACA_LIB_DIR="${MACA_PATH:-/opt/maca}/lib"
  mkdir -p "${CUDA_SHIM_DIR}/lib64"
  ln -sfn "${MACA_LIB_DIR}/libruntime_cu.so" "${CUDA_SHIM_DIR}/lib64/libcudart.so"
  export LD_LIBRARY_PATH="${MACA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  export CUDA_HOME="${CUDA_SHIM_DIR}"
elif command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  export CUDA_PATH="${CUDA_HOME}"
fi

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m pytest \
  python/sglang/jit_kernel/tests/test_fused_norm_rope_v2_bf16.py \
  -v -s "$@"

