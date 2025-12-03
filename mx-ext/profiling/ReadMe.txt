一、Profling方式
1、Complete profiling
1.1、设置环境变量：按需调整下列环境变量设置并执行命令'source profiling_env.sh'
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

1.2、执行profiling
    样例：
    python3 -m sglang.bench_offline_throughput  --dataset-name random  --dataset-path=/mnt/data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --profile  --random-input-len 128 --random-output-len 128  --trust-remote-code --dtype bfloat16 \
    --model-path /mnt/data/DeepSeek-R1-BF16-tp32 --tp 1  --num-prompts 32

2、Batch forward profiling
2.1、设置环境变量：按需调整下列环境变量设置并执行命令'source profiling_env.sh'
    # The directory to save trace files
    export SGLANG_TORCH_PROFILER_DIR=/pde_ai/share/wzp/profiling
    # enable batch forward profiling. DONT enable when profiling with '--profile'
    export MX_ENABLE_BATCH_FORWARD_PROFILE=True

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

2.2、执行profiling
    样例：
    python3 -m sglang.bench_offline_throughput  --dataset-name random  --dataset-path=/mnt/data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --random-input-len 128 --random-output-len 128  --trust-remote-code --dtype bfloat16 \
    --model-path /mnt/data/DeepSeek-R1-BF16-tp32 --tp 1  --num-prompts 32  

3、Model runner forward profiling
3.1、设置环境变量：按需调整下列环境变量设置并执行命令'source profiling_env.sh'
    # The directory to save trace files
    export SGLANG_TORCH_PROFILER_DIR=/pde_ai/share/wzp/profiling
    # enable model runner forward profiling. DONT enable when profiling with '--profile'
    export MX_ENABLE_MODEL_FORWARD_PROFILE=False

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

3.2、执行profiling
    样例：
    python3 -m sglang.bench_offline_throughput  --dataset-name random  --dataset-path=/mnt/data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --random-input-len 128 --random-output-len 128  --trust-remote-code --dtype bfloat16 \
    --model-path /mnt/data/DeepSeek-R1-BF16-tp32 --tp 1  --num-prompts 32

二、Trace分析
1. 分别将相同场景A100/C500的torch profile json文件放至a100/data， c500/data;
2. 在当前路径 执行 ./run.sh case_name, 需要给一个参数，用来命名最终生成的excel文件。
3. 每次执行记得把目录下的excel都删掉

注意：torch profile时，只能采集CUDA信息，不能包含CPU信息，否则无法正常分析。