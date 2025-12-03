# The directory to save trace files
export SGLANG_TORCH_PROFILER_DIR=/pde_ai/share/wzp/profiling

# Enable MX custom profiling settings
export MX_ENABLE_CUSTOM_PROFILE=True

# profile activities settings
# profile cuda only
export MX_PROFILE_MODE=cuda
# profile cpu only
#export MX_PROFILE_MODE=cpu
# profile both cuda and cpu
#export MX_PROFILE_MODE=xpu

# enable tensor shapes recording
export MX_PROFILE_RECORD_SHAPES=False

# enable memory profiling
export MX_PROFILE_MEMORY=False

# enable with stack profiling
export MX_PROFILE_WITH_STACK=True

# enable batch forward profiling. DONT enable when profiling with '--profile'
export MX_ENABLE_BATCH_FORWARD_PROFILE=False

# enable model runner forward profiling. DONT enable when profiling with '--profile'
export MX_ENABLE_MODEL_FORWARD_PROFILE=False