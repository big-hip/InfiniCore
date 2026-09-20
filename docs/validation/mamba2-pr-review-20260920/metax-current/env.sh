export MAMBA575_ROOT=/data/mamba575-review
export INFINI_ROOT="$MAMBA575_ROOT/runtime"
export PATH="$MAMBA575_ROOT/venv/bin:$MAMBA575_ROOT/tools/bin:/opt/conda/bin:/opt/maca/mxgpu_llvm/bin:$PATH"
export PYTHON="$MAMBA575_ROOT/venv/bin/python"
export XMAKE_ROOT=y
export XMAKE_PROGRAM_DIR="$MAMBA575_ROOT/tools/share/xmake"
export XMAKE_GLOBALDIR="$MAMBA575_ROOT/xmake-global"
export XMAKE_PKG_INSTALLDIR="$MAMBA575_ROOT/xmake-packages"
export MACA_PATH=/opt/maca MACA_HOME=/opt/maca MACA_ROOT=/opt/maca
export CPLUS_INCLUDE_PATH=/opt/conda/include/python3.12
export INFINILM_CXX11_ABI=1
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:/opt/maca/lib:/opt/conda/lib/python3.12/site-packages/torch/lib:/opt/thirdparty/lib"
export PYTHONPATH="$MAMBA575_ROOT/core/python:$MAMBA575_ROOT/lm/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_TF32_OVERRIDE=0 INFINIOP_METAX_ALLOW_TF32=0
export INFINILM_MAMBA2_MODEL="$MAMBA575_ROOT/models/mamba2-130m"
export INFINILM_MAMBA2_TP=1 INFINILM_MAMBA2_GRAPH=1
