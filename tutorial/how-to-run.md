这一页说明如何使用SGLang框架推理LLM。建议使用SGLang Docker环境，创建容器时添加模型目录映射：-v /pde_ai:/pde_ai，测试工程在/workspace/tutorial下。

目前SGLang已支持的模型包括：DeepSeek-R1-BF16, DeepSeek-R1-W8A8, DeepSeek-R1-Distill-Qwen-1.5B。

## 启动容器

```shell
docker run -itd --privileged=true --device=/dev/dri --device=/dev/mxcd --group-add video --name ${container_name} --device=/dev/mem --net=host --uts=host --ipc=host --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size '100gb' --ulimit memlock=-1 -v /mnt/hdd/SGLang:/software ${images_id} /bin/bash
```
其中\${container_name}为指定的容器名，/mnt/hdd/SGLang:/software为指定的挂载目录，需要自己指定目录，\${images_id}为指定的镜像ID，如果需要多机启动，则基于该镜像在多个机器上使用如上指令启动相应的容器。

## sglang目录结构及说明
```
.
├── code
│   ├── run_ceval_client.py
│   ├── openai_completion_client.py
│   ├── openai_chatcompletion_client.py
│   ├── bench_sglang.py
│   
├── dataset
│   ├── ceval_val_cmcc.jsonl
│   ├── ShareGPT_V3_unfiltered_cleaned_split.json
```
code下存放的是测试代码和脚本，dataset下存放相关数据集的文件，上述的模型中DeepSeek-R1-BF16，DeepSeek-R1-Distill-Qwen-1.5B请至相关官网下载， DeepSeek-R1-W8A8模型需要根据沐曦发布的文档（可参考https://developer.metax-tech.com/doc/278中"3.1.2 W8A8 模型转换"部分）自行量化模型。

## 启动server

1. 设置环境变量

如果是单机只需要在一台机器上的容器中执行，如果多机该操作需要在所有的容器中都进行一次。
```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
```
如果多机环境，需要设置以下环境变量
```shell
export GLOO_SOCKET_IFNAME=网口名
```
对于GLOO_SOCKET_IFNAME环境变量，我们需要在宿主机上执行 ifconfig -a 指令找到与该宿主机ip地址对应的网口名，然后设置到该环境变量上。


2. 启动server

对于DeepSeek-R1-BF16全量模型，我们以4机32卡为例：当前运行建议按照tp切分为32。
```
python3 -m sglang.launch_server --model-path ${Model_path} --tp 32 --dist-init-addr 100.79.153.153:5000 --nnodes 4 --node-rank 0 --trust-remote-code --disable-cuda-graph

python3 -m sglang.launch_server --model-path ${Model_path} --tp 32 --dist-init-addr 100.79.153.153:5000 --nnodes 4 --node-rank 1 --trust-remote-code --disable-cuda-graph

python3 -m sglang.launch_server --model-path ${Model_path} --tp 32 --dist-init-addr 100.79.153.153:5000 --nnodes 4 --node-rank 2 --trust-remote-code --disable-cuda-graph

python3 -m sglang.launch_server --model-path ${Model_path} --tp 32 --dist-init-addr 100.79.153.153:5000 --nnodes 4 --node-rank 3 --trust-remote-code --disable-cuda-graph
```
其中--tp 32 表示tp并行的切分数量为32。

--dist-init-addr 100.79.153.153:5000是指定主节点的ip和端口号（可以默认成5000），其他三机需和主节点保持一致。

--nnodes 4 表示节点数量，--node-rank 0 表示当前机器所属的节点索引，需要注意的是必须主节点先启动，然后其他节点才能启动。

对于DeepSeek-R1-W8A8量化模型，我们可以使用16卡即可运行，我们用2机16卡为例：当前运行建议按照tp切分为16。 
```
python3 -m sglang.launch_server --model-path ${Model_path} --tp 16 --dist-init-addr 100.79.153.153:5000 --nnodes 2 --node-rank 0 --trust-remote-code --disable-cuda-graph

python3 -m sglang.launch_server --model-path ${Model_path} --tp 16 --dist-init-addr 100.79.153.153:5000 --nnodes 2 --node-rank 1 --trust-remote-code --disable-cuda-graph
```

对于DeepSeek-R1-Distil-Qwen-1.5B模型，我们使用单卡即可运行，我们以单卡为例：

python3 -m sglang.launch_server --model-path ${Model_path} --tp 1 --trust-remote-code --disable-cuda-graph

## benchmark throughput 性能测试
当所有server都启动之后，需要在主节点容器内重新起一个终端，进行benchmark throughput 性能测试。
```
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input-len ${input-len} --random-output-len ${output-len} --random-range-ratio 1.0 --dataset-path ./dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompt ${batch_size}
```
其中 \${input-len} 表示输入长度，\${output-len} 表示输出长度，\${batch_size} 指定输入批次数量。


## mmlu 精度测试

如果使用mmlu数据集进行精度测试，需要准备data数据，请从https://people.eecs.berkeley.edu/~hendrycks/data.tar下载，解压并拷贝至dataset路径。此外，还需要从https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken下载"cl100k_base.tiktoken"文件，放到容器内任意位置。

1. 安装依赖
```shell
pip install blobfile
export TIKTOKEN_CACHE_DIR=${path}
```
${path} 表示 "cl100k_base.tiktoken" 文件所在的路径，不需要包含文件名。

2. 执行
```
python code/bench_sglang.py --nsub 10 --data_dir ./dataset/data
```
其中--nsub是指定测试问题的数量，最多是60，也可以不指定默认是60。

## ceval 精度测试
1. 安装依赖
```shell
pip install eval-type-backport
```
2. 执行
使用run_ceval_client.py进行精度测试，使用的测试命令行如下：
```
python code/run_ceval_client.py --model ${Model_path} --test_jsonl ./dataset/ceval_val_cmcc.jsonl --batch_size 64
```
其中-model ${Model_path}为model所在的路径，--test_jsonl ./dataset/ceval_val_cmcc.jsonl为ceval数据集的路径

需要注意的是该脚本对接的server的端口号是8000， 所以我们需要在启动sglang server时加上 --port 8000，示例如下：
```
python -m sglang.launch_server --model ${Model_path} --tp 32 --dist-init-addr 100.79.153.153:5000 --nnodes 4 --node-rank 0 --trust-remote-code --disable-cuda-graph --port 8000 
```