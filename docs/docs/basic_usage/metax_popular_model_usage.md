# MetaX LLM 模型基于 SGLang 的推理部署用户手册


> **适用范围**
> 本手册规定 MetaX MXC5XX、MXC600 和 MXC600U GPU 平台上 SGLang 大语言模型的推理部署与验证方法。
>
> **版本基线**
> 本手册使用与目标 SGLang 版本匹配的容器镜像。模型、GPU 平台、并行配置、环境变量和测试方法以各章节列出的配置为准。
>
> **读者对象**
> 本手册面向负责 SGLang 环境部署、模型服务启动和性能或精度验证的技术人员。

---

# 目录

1. [软硬件需求](#1-软硬件需求)
   1.1 [硬件需求](#11-硬件需求)
   1.2 [软件需求](#12-软件需求)
2. [环境搭建](#2-环境搭建)
   2.1 [基础环境配置](#21-基础环境配置)
   2.2 [网口与IB配置](#22-网口与ib配置)
   2.3 [GPU拓扑与设备顺序](#23-gpu拓扑与设备顺序)
   2.4 [环境检查和设置](#24-环境检查和设置)
   2.5 [SGLang 版本确认](#25-sglang-版本确认)
3. [DeepSeek 系列推理部署](#3-deepseek-系列推理部署)
4. [GLM 系列推理部署](#4-glm-系列推理部署)
5. [MiniMax-M3 系列推理部署](#5-minimax-m3-系列推理部署)
6. [Kimi 系列推理部署](#6-kimi-系列推理部署)
7. [Qwen 系列推理部署](#7-qwen-系列推理部署)
8. [MiMo 系列推理部署](#8-mimo-系列推理部署)
9. [推理测试](#9-推理测试)

---

部署和验证按照以下流程执行：

1. **环境准备**：确认硬件、驱动、Docker、镜像、网络、IB 和系统性能设置；
2. **模型准备**：准备目标模型权重，并确认模型精度与部署方式；
3. **模型部署**：根据模型章节选择对应启动参数；
4. **功能验证**：确认 Server 端口启动并完成基础 `/generate` 请求；
5. **性能与精度测试**：执行随机性能测试、C-Eval 或 MMLU；
6. **结果记录**：记录镜像、MACA、Driver、SGLang、模型路径、并行参数和测试结果。

---

# 1. 软硬件需求

## 1.1 硬件需求

本手册统一适用于 MetaX **MXC5XX、MXC600、MXC600U** 系列 GPU。部署前应确认：

- GPU 数量满足目标模型及目标并行配置；
- GPU 显存满足模型权重、KV Cache、CUDA Graph/Graph Cache、Runtime Buffer 等占用；
- 多卡/多机部署时 PCIe、GPU 互联和网络/IB 链路正常；
- 主机内存和模型所在磁盘空间充足。

模型权重显存可按以下公式估算：

```text
1. FP16/BF16（半精度）≈ 参数量 × 2 × 1.2
2. INT8/W8A8           ≈ 参数量 × 1.1
```

KV Cache 还会额外占用显存。以 BF16 为例，可使用如下形式理解其主要影响因素：

```text
num_hidden_layers * head_dim * seq_len * bs * 2 * precision
```

上下文长度 `seq_len`、批处理大小、层数和精度均影响 KV Cache 占用。SGLang 会根据显存池和上下文长度预留 Block；GPU 卡数必须同时依据模型权重、KV Cache、并行配置和运行时缓冲区确定。

### 1.1.1 支持模型与部署平台

SGLang 根据显存池和上下文长度预留 KV Cache Block。上下文越长、并发越高，所需显存越大。下表的卡数仅由 `TP × PP` 计算，不计入 DP；实际部署仍须结合显存、拓扑和目标负载确认。

**MXC600U 平台（FP8）**

| 模型配置 | 并行配置 | 所需卡数 | 权重精度 |
| --- | --- | ---: | --- |
| DeepSeek-V4-Flash-FP8 | TP8DP8 | 8 | FP8 |
| DeepSeek-V4-Flash-FP8 | TP16DP2MTP112 | 16 | FP8 |
| DeepSeek-V4-Pro-FP8 | TP8PP4 | 32 | FP8 |
| DeepSeek-V4-Flash-DSpark-FP8 | TP8DP8 | 8 | FP8 |
| GLM-5.1-FP8 | TP16DP4MTP314 | 16 | FP8 |
| GLM-5.2-FP8 | TP16DP4MTP314 | 16 | FP8 |
| Kimi-K2.5-FP8 | TP8PP4 | 32 | FP8 |
| Qwen3.5-397B-FP8 | TP8DP8 | 8 | FP8 |
| MiniMax-M3-FP8 | TP8DP8 | 8 | FP8 |

**MXC5XX/MXC600 平台（W8A8 与 BF16）**

| 模型配置 | 并行配置 | 所需卡数 | 权重精度 |
| --- | --- | ---: | --- |
| DeepSeek-R1-W8A8 | TP16DP4MTP213 | 16 | W8A8 |
| DeepSeek-V3.2-W8A8 | TP16DP2 | 16 | W8A8 |
| DeepSeek-V4-Pro-W8A8 | TP4PP8 | 32 | W8A8 |
| DeepSeek-V4-Flash-W8A8 | TP8MTP213 | 8 | W8A8 |
| GLM-5.2-W8A8 | TP16DP4MTP314 | 16 | W8A8 |
| GLM-5.2-W8A8 | TP8PP2CP8 | 16 | W8A8 |
| MiMo-V2-Flash-W8A8 | TP16DP4 | 16 | W8A8 |
| MiniMax-M3-W8A8 | TP16DP2 | 16 | W8A8 |
| Kimi-K2.5/K2.6-W8A8 | TP32DP4 | 32 | W8A8 |
| Qwen3.5-397B-W8A8 | TP16 | 16 | W8A8 |
| Qwen3.6-27B | TP2 | 2 | BF16 |
| Qwen3.6-35B-A3B | TP2 | 2 | BF16 |

### 1.1.2 统一硬件说明

本手册统一适用于 MXC5XX、MXC600、MXC600U。不同 GPU 系列使用对应的环境变量和模型启动参数。

### 1.1.3 GPU 状态检查

在宿主机执行：

```shell
mx-smi
```

重点确认：GPU 数量、GPU 型号、显存、Driver、MACA 版本和 GPU 状态是否正常。

查看 GPU 拓扑：

```shell
mx-smi topo -m
```

多机部署还需要确认网络与 IB 状态：

```shell
ifconfig -a
ibstat
```

---

## 1.2 软件需求

### 1.2.1 获取 MetaX SGLang 基础镜像

在已安装 Docker 的服务器上，从 MetaX 官方镜像仓库获取与目标部署环境匹配的 SGLang 基础镜像。多节点部署时，所有节点必须使用完全相同的镜像地址和版本。

1. 打开 [MetaX 官方 Docker 镜像仓库](https://developer.metax-tech.com/softnova/docker)。
2. 根据实际部署 GPU 系列选择对应平台，在 `AI` 分类下选择推理框架中的 `SGLANG`。
3. 根据部署需求选择镜像架构、操作系统、Python 版本和 SGLang 框架版本。
4. 找到目标镜像后，复制页面提供的完整拉取命令，并在本地终端执行。

创建容器时使用该镜像；多节点部署的所有节点必须使用相同镜像。

### 1.2.2 软件版本检查

进入容器后执行以下命令确认软件版本：

```shell
python3 -c "import sglang; print(sglang.__version__); print(sglang.__file__)"
python3 -V
mx-smi
```

---

# 2. 环境搭建

本章流程适用于 MXC5XX、MXC600 和 MXC600U。设备顺序、环境变量和模型启动参数须与目标硬件和模型配置一致。

## 2.1 基础环境配置

### 2.1.1 多节点环境一致性

多节点部署前，在每台宿主机完成以下一致性检查：

1. 所有节点使用第 1.2 节选定的同一镜像地址和版本；建议使用 Digest 固定镜像版本。
2. 所有节点的 Driver、MACA、Docker 运行环境与目标镜像兼容；容器内的 `python3 -V`、`mx-smi` 和 `sglang.__version__` 应保持一致。
3. 模型、草稿模型和数据集目录在所有节点的容器内使用相同路径；宿主机路径可不同，但容器内挂载路径必须一致。
4. 各节点使用可互通的计算网络，节点 0 的分布式初始化地址和端口对其他节点可达。
5. 所有节点使用相同的并行参数、`--nnodes` 和服务配置，仅 `--node-rank` 按节点编号区分。

容器挂载参数在第 2.1.2 节的 `docker run` 命令中配置。模型或数据集位于大容量磁盘时，使用 `-v <宿主机目录>:<容器内目录>` 映射到上述统一的容器内路径。

### 2.1.2 创建容器

使用以下容器创建模板：

```shell
# $container_name：容器名称
# $image_id：第 1.2 节指定的镜像 ID 或 Tag

docker run -it \
  --device=/dev/dri \
  --device=/dev/mxcd \
  --device=/dev/infiniband \
  --privileged=true \
  --group-add video \
  --name $container_name \
  --device=/dev/mem \
  --network=host \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --shm-size '100gb' \
  --ulimit memlock=-1 \
  -v /usr/local/:/usr/local/ \
  -v /宿主机模型目录:/容器内模型目录 \
  -v /宿主机数据集目录:容器内数据集目录 \
  $image_id /bin/bash
```

其中：

- `--security-opt seccomp=unconfined`：解除 seccomp 限制，避免出现线程权限不足报错；
- `--device=/dev/infiniband` 用于需要 IB 的环境；
- `--network=host`：用于多节点分布式通信和服务端口访问；
- `--shm-size '100gb'` 为容器提供较大的共享内存；
- 模型和数据集必须映射至容器内可访问的存储目录。

---

## 2.2 网口与IB配置

多机部署或使用 RDMA 时，保留第 2.1.2 节中的 `--device=/dev/infiniband` 和 `--network=host`。节点间通信使用的网口与 IB 网卡应以服务器实际拓扑为准；单机部署通常不需要设置本节变量。

### 2.2.1 查看计算网口

在每个节点的宿主机或容器内执行以下命令，查看网卡名称和 IP 地址：

```shell
ifconfig -a

# 未安装 ifconfig 时，可使用以下命令
ip -br address
```

找到与当前节点计算 IP 对应的网卡名称，并将该网卡用于节点间通信。必要时，可通过 `ip route get <MASTER_IP>` 确认访问主节点实际使用的网卡。

将查到的网卡名称设置为以下环境变量。所有节点应按各自实际网口设置，不应照抄示例名称。

```shell
export GLOO_SOCKET_IFNAME=<网卡名>
export MCCL_SOCKET_IFNAME=<网卡名>
```

例如，若计算网口为 `ens115f0`：

```shell
export GLOO_SOCKET_IFNAME=ens115f0
export MCCL_SOCKET_IFNAME=ens115f0
```

### 2.2.2 IB HCA 配置

在每个节点执行以下命令，查看 IB 网卡状态：

```shell
ibstat

# 可选：查看 IB HCA 与网口的对应关系
ibdev2netdev
```

选择名称符合 `mlx5_<数字>`、状态为 `State: Active` 且用于 RDMA 通信的 HCA，并设置 `MCCL_IB_HCA`。多个 HCA 使用英文逗号分隔：

```shell
export MCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
```

未确认 `mlx5_bond_*` 为 RDMA HCA 时，不得将其加入该变量。未配置或未使用 RDMA 的环境不要设置 `MCCL_IB_HCA`。

### 2.2.3 多机连通性检查

启动前确认节点间路由和防火墙策略允许分布式初始化端口通信：

```shell
ping <peer_ip>
ip route get <MASTER_IP>
```

节点 0 启动并监听 `--dist-init-addr <MASTER_IP>:<DIST_PORT>` 后，可从其他节点执行以下命令验证端口可达：

```shell
nc -zv <MASTER_IP> <DIST_PORT>
```

未安装 `nc` 时，使用等效工具验证端口连通性。分布式初始化失败时，检查 IP、网卡、端口、`nnodes`、`node-rank` 和相关环境变量配置。

---

## 2.3 GPU拓扑与设备顺序

执行：

```shell
mx-smi topo -m
```

GPU 互联顺序由服务器拓扑决定。设备编号未按互联组连续排列时，按 `mx-smi topo -m` 的输出设置可见设备顺序。例如，当拓扑显示 `0,1,2,3` 连通且 `4,5,6,7` 连通时，可直接进行下一步，如果不是按顺序排列的，那就需要设置环境变量指定GPU的顺序，例如：
```shell
export MACA_VISIBLE_DEVICES=7,5,1,3,4,0,6,2
```

> 示例值不代表实际部署配置。MXC5XX、MXC600 和 MXC600U 的设备顺序必须以当前服务器的 `mx-smi topo -m` 输出为准。

---

## 2.4 环境检查和设置

在启动模型服务前，在宿主机执行环境检查。该脚本检查 CPU 电源模式、GPU 电源模式、PCIe ACS/APSM、NUMA 和网络接口状态，并输出检查结果。

环境检查脚本位于 [`python/sglang/test/metax/mx_env_chk.py`](../../../python/sglang/test/metax/mx_env_chk.py)。脚本默认会在检测到不符合要求的项目时尝试修改系统设置；请先执行只检查模式，确认结果后再执行完整设置。

> 执行完整设置需要 root 权限，且可能修改 CPU 电源模式、PCIe ACS 和 GPU 电源模式。请在业务低峰期执行，并确认目标服务器允许调整这些系统设置。

### 2.4.1 只检查当前环境

```shell
# 在 SGLang 仓库根目录执行；该命令不修改系统设置
python3 python/sglang/test/metax/mx_env_chk.py --check-only
```

检查结果全部符合要求时，脚本返回 `0`；发现需要调整的项目时返回 `1`。根据输出确认 CPU、GPU、PCIe、NUMA 和网络接口状态是否符合部署要求。

### 2.4.2 执行环境设置

仅在完成检查并确认允许调整系统设置后执行以下命令：

```shell
sudo python3 python/sglang/test/metax/mx_env_chk.py
```

脚本完成设置后会再次验证。验证成功返回 `0`，验证失败返回 `2`。如验证失败，请保留完整输出并结合服务器的 BIOS、驱动版本、网卡和 GPU 拓扑继续排查。

### 2.4.3 查看帮助与 PCIe APSM 信息

```shell
# 查看参数说明
python3 python/sglang/test/metax/mx_env_chk.py --help

# 在检查或设置流程中显示 PCIe APSM 信息
python3 python/sglang/test/metax/mx_env_chk.py --check-only --apsm
```

多机部署时，每台宿主机均应执行一次环境检查。网络接口名称、通信网卡选择和 GPU 设备顺序仍以第 2.2 节和第 2.3 节的实际服务器检查结果为准。

---

## 2.5 SGLang 版本确认

完成容器创建后、启动模型服务前，在每个节点记录运行环境基线信息。该记录用于确认节点间的软件版本和环境变量一致，并为后续问题排查保留依据。

```shell
ENV_RECORD_DIR="/tmp/sglang_env_$(hostname)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ENV_RECORD_DIR"

mx-smi > "$ENV_RECORD_DIR/mx-smi.txt"
python3 -V > "$ENV_RECORD_DIR/python.txt" 2>&1
pip show sglang torch sgl-kernel flashinfer-python > "$ENV_RECORD_DIR/packages.txt" 2>&1
env | grep -E 'MACA|MCCL|GLOO|TRITON|TORCH' | sort > "$ENV_RECORD_DIR/env.txt"
python3 -c "import sglang; print(sglang.__file__); print(getattr(sglang,'__version__','unknown'))" > "$ENV_RECORD_DIR/sglang.txt" 2>&1
```

核对各节点记录中的 SGLang、Python、Driver、MACA 和镜像版本。确认 SGLang 版本与本次部署选择的镜像版本一致，且多节点间不存在版本差异；如不一致，停止模型部署并核对容器镜像和 Python 模块加载路径。

---

# 3. DeepSeek 系列推理部署

本章覆盖 DeepSeek-R1-W8A8、DeepSeek-V3.2-W8A8、DeepSeek-V4-Flash-W8A8、DeepSeek-V4-Pro-W8A8 以及 DeepSeek-V4 的 FP8 配置。按“模型准备 → 环境变量设置 → 服务部署 → 第 9 章验证”的顺序执行。

## 3.1 模型准备

确认模型目录包含完整权重、分词器和配置文件，且精度与目标启动命令一致。DeepSeek-R1-W8A8 的 NEXTN 推测解码配置还必须提供匹配的 NextN draft 模型目录。

| 模型配置 | 下载地址 |
| --- | --- |
| DeepSeek-R1-W8A8 | [MetaX：DeepSeek-R1-0528-W8A8](https://www.modelscope.cn/models/metax-tech/DeepSeek-R1-0528-W8A8) |
| DeepSeek-V3.2-W8A8 | [MetaX：DeepSeek-V3.2-W8A8](https://www.modelscope.cn/models/metax-tech/DeepSeek-V3.2-W8A8) |
| DeepSeek-V4-Flash-W8A8、DSpark-FP8 | [DeepSeek：DeepSeek-V4-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash) |
| DeepSeek-V4-Flash-FP8 | [SGLang：DeepSeek-V4-Flash-FP8](https://www.modelscope.cn/models/sgl-project/DeepSeek-V4-Flash-FP8) |
| DeepSeek-V4-Pro-W8A8、FP8 | [DeepSeek：DeepSeek-V4-Pro](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro) |

## 3.2 环境变量设置

MXC5XX/MXC600 配置使用以下通用变量；

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

MXC600U 的 FP8 配置使用以下通用变量：

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

各模型的专用变量在对应的部署配置中单独列出；仅执行与所选模型和并行配置对应的变量。

## 3.3 服务部署

非 FP8 配置的节点数按所需 GPU 卡数确定：8 卡及以下在单机执行；16 卡在两台主机执行；32 卡在四台主机执行。多机时，所有节点保持相同的模型和并行参数，`<MASTER_IP>` 填写节点 0 的通信 IP，`--node-rank` 按节点编号设置。

### 3.3.1 DeepSeek-V4-Pro-W8A8（MXC5XX/MXC600，TP4PP8）

**模型准备**：下载 DeepSeek-V4-Pro-W8A8 模型包，确认模型目录包含完整权重、分词器和配置文件。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MX_ENABLE_DEQUANT_BF16=true
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
```
**启动服务**：该配置使用 32 张 GPU，按下方命令在四台主机启动服务。

该配置需要 32 张 GPU。以四节点部署为例，在四台主机的同一容器环境中分别执行以下命令：所有节点使用相同的模型、并行和通信参数，仅将 `--node-rank` 依次替换为 `0`、`1`、`2`、`3`；`<MASTER_IP>` 填写节点 0 的通信 IP。

```shell
python3 -m sglang.launch_server \
  --trust-remote-code \
  --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache \
  --tp-size 4 --pp-size 8 \
  --mem-fraction-static 0.84 \
  --kv-cache-dtype bf16 \
  --cuda-graph-max-bs-decode 8 \
  --chunked-prefill-size 8192 \
  --swa-full-tokens-ratio 0.3 \
  --dist-init-addr <MASTER_IP>:5000 \
  --nnodes 4 \
  --node-rank <0-3> \
  --port 30000
```

### 3.3.2 DeepSeek-V4-Flash-W8A8（MXC5XX/MXC600，TP8MTP213）

**模型准备**：下载 DeepSeek-V4-Flash-W8A8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```
**启动服务**：该配置使用 8 张 GPU，按下方命令在单机启动服务。

该配置使用 8 张 GPU，在单机执行以下命令。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache \
  --tp-size 8 --mem-fraction-static 0.75 \
  --kv-cache-dtype bf16 --cuda-graph-max-bs-decode 64 \
  --chunked-prefill-size 8192 --swa-full-tokens-ratio 0.3 \
  --speculative-algorithm EAGLE --speculative-num-steps 2 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
  --quantization w8a8_int8
```

### 3.3.3 DeepSeek-R1-W8A8（MXC5XX/MXC600，TP16DP4MTP213）

**模型准备**：下载 DeepSeek-R1-W8A8 模型包和匹配的 NextN draft 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```
**启动服务**：该配置使用 16 张 GPU，按下方命令在两台主机启动服务。

该配置使用 16 张 GPU、NEXTN 推测解码和对应的 NextN draft 模型。在两台主机上执行以下命令，并分别设置 `--node-rank 0` 和 `--node-rank 1`。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache \
  --attention-backend flashinfer \
  --tp-size 16 \
  --disable-overlap-schedule \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path <draft 模型目录> \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --quantization w8a8_int8 \
  --dist-init-addr <MASTER_IP>:5000 \
  --nnodes 2 \
  --node-rank <0-1> \
  --port 30000
```

### 3.3.4 DeepSeek-V3.2-W8A8（MXC5XX/MXC600，TP16DP2）

**模型准备**：下载 DeepSeek-V3.2-W8A8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
export MX_ENABLE_FLASH_MLA_OPT=1
```
**启动服务**：该配置使用 16 张 GPU，按下方命令在两台主机启动服务。

该配置使用 16 张 GPU。在两台主机上执行以下命令，并分别设置 `--node-rank 0` 和 `--node-rank 1`。启动前设置 DeepSeek-V3.2-W8A8 的 Flash MLA 追加环境变量。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --attention-backend flashinfer \
  --tp-size 16 --dp-size 2 --enable-dp-attention --enable-dp-lm-head \
  --mem-fraction-static 0.90 --quantization w8a8_int8 \
  --dist-init-addr <MASTER_IP>:5000 \
  --nnodes 2 \
  --node-rank <0-1> \
  --port 30000
```

### 3.3.5 DeepSeek-V4-Flash-FP8（MXC600U）

**模型准备**：下载 DeepSeek-V4-Flash-FP8 模型包；两种并行配置使用相同权重。  
**环境变量设置**：两种并行配置均先执行以下完整基础环境变量；TP8DP8 再执行其专用变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export ENABLE_MCTLASS_FUSED_MOE=0
export ENABLE_MCTLASS_FUSED_MOE_PYTHON_API=1
export SGLANG_OPT_FP8_WO_A_GEMM=1
export SGLANG_DSV4_INDEX_CACHE_TYPE=fp8
export SGLANG_OPT_USE_FUSED_STORE_CACHE=False
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=deep_gemm
export SGLANG_OPT_USE_COMPRESSOR_V2=False
export SGLANG_ENABLE_JIT_DEEPGEMM=True
```
**启动服务**：选择下方 TP8DP8 或 TP16DP2MTP112 中的一种配置启动服务。

#### 3.3.5.1 TP8DP8

TP8DP8 还需追加：

```shell
export MACA_KERNEL_TIMEOUT=60000
export MCCL_MAX_NCHANNELS=8
```

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --attention-backend flashinfer \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --mem-fraction-static 0.88 --cuda-graph-max-bs-decode 128 \
  --max-running-requests 128 --chunked-prefill-size 8192
```

#### 3.3.5.2 TP16DP2MTP112

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --attention-backend flashinfer \
  --tp-size 16 --dp-size 2 --enable-dp-attention --enable-dp-lm-head \
  --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 128 \
  --max-running-requests 128 --chunked-prefill-size 8192 \
  --speculative-algorithm EAGLE --speculative-num-steps 1 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 2
```

### 3.3.6 DeepSeek-V4-Pro-FP8（MXC600U，TP8PP4）

**模型准备**：下载 DeepSeek-V4-Pro-FP8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export ENABLE_MCTLASS_FUSED_MOE=0
export ENABLE_MCTLASS_FUSED_MOE_PYTHON_API=1
export SGLANG_OPT_FP8_WO_A_GEMM=1
export SGLANG_DSV4_INDEX_CACHE_TYPE=fp8
export SGLANG_OPT_USE_FUSED_STORE_CACHE=False
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=deep_gemm
export SGLANG_OPT_USE_COMPRESSOR_V2=False
export SGLANG_ENABLE_JIT_DEEPGEMM=True
export MACA_KERNEL_TIMEOUT=60000
export MCCL_MAX_NCHANNELS=8
```
**启动服务**：该配置使用 32 张 GPU，执行下方命令启动服务。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --attention-backend flashinfer \
  --tp-size 8 --pp-size 4 --mem-fraction-static 0.85 \
  --cuda-graph-max-bs-decode 64 --max-running-requests 64 \
  --chunked-prefill-size 8192
```

### 3.3.7 DeepSeek-V4-Flash-DSpark-FP8（MXC600U，TP8DP8）

**模型准备**：下载 DeepSeek-V4-Flash-DSpark-FP8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
export MX_ENABLE_DEQUANT_BF16=true
export TILELANG_DISABLE_CACHE=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_ENABLE_JIT_DEEPGEMM=True
export SGLANG_DSV4_INDEX_CACHE_TYPE=fp8
```
**启动服务**：该配置使用 8 张 GPU，执行下方命令启动服务。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --disable-chunked-prefix-cache \
  --tp-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
  --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 128 \
  --max-running-requests 128 --chunked-prefill-size 8192 \
  --speculative-algorithm DSPARK
```

---

# 4. GLM 系列推理部署

GLM 系列按“模型准备 → 环境变量设置 → 服务部署 → 第 9 章验证”的顺序执行。FP8 配置使用 MXC600U；W8A8 配置使用 MXC5XX 或 MXC600。GLM-5.2-W8A8 的两种并行配置分别对应不同并行策略，单次部署仅选择其中一种。

## 4.1 模型准备

下载与目标精度一致的模型包，并确认模型目录包含完整权重、分词器和配置文件。

| 模型配置 | 下载地址 |
| --- | --- |
| GLM-5.1-FP8 | [ZhipuAI/GLM-5.1-FP8](https://www.modelscope.cn/models/ZhipuAI/GLM-5.1-FP8) |
| GLM-5.2-FP8 | [ZhipuAI/GLM-5.2-FP8](https://modelscope.cn/models/ZhipuAI/GLM-5.2-FP8/summary) |
| GLM-5.2-W8A8 | [ZhipuAI：GLM-5.2-FP8（官方基座）](https://modelscope.cn/models/ZhipuAI/GLM-5.2-FP8/summary) |

## 4.2 环境变量设置

MXC5XX/MXC600 配置使用以下通用变量：

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

MXC600U 的 FP8 配置使用以下通用变量：

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

各模型的专用变量在对应部署配置中单独列出；仅执行与所选模型和并行配置对应的变量。

## 4.3 服务部署

### 4.3.1 GLM-5.1-FP8（MXC600U，TP16DP4MTP314）

**模型准备**：下载 GLM-5.1-FP8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export ENABLE_MCTLASS_FUSED_MOE=0
export ENABLE_MCTLASS_FUSED_MOE_PYTHON_API=1
export SGLANG_ENABLE_JIT_DEEPGEMM=True
export MACA_KERNEL_TIMEOUT=60000
export SGLANG_OPT_FP8_WO_A_GEMM=1
export SGLANG_DSV4_INDEX_CACHE_TYPE=fp8
export SGLANG_OPT_USE_FUSED_STORE_CACHE=False
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=deep_gemm
export MX_ENABLE_FLASH_MLA_OPT=False
export MACA_LAUNCH_MODE=0
export MACA_DIRECT_DISPATCH=1
```
**启动服务**：该配置使用 16 张 GPU，执行下方命令启动服务。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --cuda-graph-max-bs-decode 64 --max-running-requests 64 \
  --chunked-prefill-size 8192 --disable-radix-cache \
  --disable-chunked-prefix-cache --tp-size 16 --dp-size 4 \
  --enable-dp-attention --enable-dp-lm-head \
  --mem-fraction-static 0.85 --speculative-algorithm EAGLE \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

### 4.3.2 GLM-5.2-FP8（MXC600U，TP16DP4MTP314）

**模型准备**：下载 GLM-5.2-FP8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export ENABLE_MCTLASS_FUSED_MOE=0
export ENABLE_MCTLASS_FUSED_MOE_PYTHON_API=1
export SGLANG_ENABLE_JIT_DEEPGEMM=True
export MACA_KERNEL_TIMEOUT=60000
export SGLANG_OPT_FP8_WO_A_GEMM=1
export SGLANG_DSV4_INDEX_CACHE_TYPE=fp8
export SGLANG_OPT_USE_FUSED_STORE_CACHE=False
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=deep_gemm
export MX_ENABLE_FLASH_MLA_OPT=False
export MACA_LAUNCH_MODE=0
export MACA_DIRECT_DISPATCH=1
```
**启动服务**：该配置使用 16 张 GPU，执行下方命令启动服务。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --cuda-graph-backend-decode=disabled --max-running-requests 128 \
  --chunked-prefill-size 8192 --disable-radix-cache \
  --disable-chunked-prefix-cache --tp-size 16 --dp-size 4 \
  --enable-dp-attention --enable-dp-lm-head \
  --mem-fraction-static 0.85 --speculative-algorithm NEXTN \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

### 4.3.3 GLM-5.2-W8A8（MXC5XX/MXC600）

#### 4.3.3.1 TP16DP4MTP314

**模型准备**：下载 GLM-5.2-W8A8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
export MX_ENABLE_FLASH_MLA_OPT=1
```
**启动服务**：该配置使用 16 张 GPU，按下方命令在两台主机启动服务。

该配置使用 16 张 GPU。在两台主机上执行以下命令，并分别设置 `--node-rank 0` 和 `--node-rank 1`。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --cuda-graph-max-bs-decode 64 --max-running-requests 64 \
  --chunked-prefill-size 4096 --disable-radix-cache \
  --disable-chunked-prefix-cache --tp-size 16 --dp-size 4 \
  --enable-dp-attention --enable-dp-lm-head --mem-fraction-static 0.89 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --quantization w8a8_int8 \
  --dist-init-addr <MASTER_IP>:5000 \
  --nnodes 2 \
  --node-rank <0-1> \
  --port 30000
```

#### 4.3.3.2 TP8PP2CP8

**模型准备**：下载 GLM-5.2-W8A8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
export MX_ENABLE_FLASH_MLA_OPT=1
```
**启动服务**：该配置使用 16 张 GPU，按下方命令在两台主机启动服务。

该配置使用 16 张 GPU。在两台主机上执行以下命令，并分别设置 `--node-rank 0` 和 `--node-rank 1`。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --cuda-graph-bs-decode 8 16 24 32 64 --max-running-requests 64 \
  --chunked-prefill-size 8192 --disable-radix-cache \
  --tp-size 8 --pp-size 2 --attn-cp-size 8 --mem-fraction-static 0.90 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split --quantization w8a8_int8 \
  --dist-init-addr <MASTER_IP>:5000 \
  --nnodes 2 \
  --node-rank <0-1> \
  --port 30000
```

---

# 5. MiniMax-M3 系列推理部署

MiniMax-M3 按“模型准备 → 环境变量设置 → 服务部署 → 第 9 章验证”的顺序执行。FP8 配置使用 MXC600U，W8A8 配置使用 MXC5XX 或 MXC600。

## 5.1 模型准备

下载与目标精度一致的模型包：FP8 使用 [MiniMax-M3-MXFP8](https://www.modelscope.cn/models/MiniMax/MiniMax-M3-MXFP8)，W8A8 使用 [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3)。

## 5.2 环境变量设置

MXC5XX/MXC600 配置使用以下通用变量：

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

MXC600U 的 FP8 配置使用以下通用变量：

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

各模型的专用变量在对应部署配置中单独列出；仅执行与所选模型和并行配置对应的变量。

## 5.3 服务部署

### 5.3.1 MiniMax-M3-FP8（MXC600U，TP8DP8）

**模型准备**：下载 MiniMax-M3-FP8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```
**启动服务**：该配置使用 8 张 GPU，执行下方命令启动服务。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --disable-chunked-prefix-cache \
  --attention-backend flashinfer --tp-size 8 --dp-size 8 \
  --enable-dp-attention --mem-fraction-static 0.90 \
  --quantization fp8 --cuda-graph-max-bs-decode 64 --max-running-requests 128
```

### 5.3.2 MiniMax-M3-W8A8（MXC5XX/MXC600，TP16DP2）

**模型准备**：下载 MiniMax-M3-W8A8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MX_ENABLE_DEQUANT_BF16=true
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
```
**启动服务**：该配置使用 16 张 GPU，按下方命令在两台主机启动服务。

该配置使用 16 张 GPU。在两台主机上执行以下命令，并分别设置 `--node-rank 0` 和 `--node-rank 1`。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --disable-chunked-prefix-cache \
  --attention-backend flashinfer --tp-size 16 --dp-size 2 \
  --enable-dp-attention --mem-fraction-static 0.7 \
  --quantization w8a8_int8 --cuda-graph-max-bs-decode 256 \
  --max-running-requests 1024 \
  --dist-init-addr <MASTER_IP>:5000 \
  --nnodes 2 \
  --node-rank <0-1> \
  --port 30000
```

---

# 6. Kimi 系列推理部署

本章按“模型准备 → 环境变量设置 → 服务部署 → 第 9 章验证”的顺序提供 Kimi-K2.5-FP8 在 MXC600U 平台上的部署配置。

## 6.1 模型准备

下载 Kimi-K2.5-FP8 或 Kimi-K2.5/K2.6-W8A8 模型包，并确认模型目录包含完整权重、分词器和配置文件。Kimi-K2.5 与 Kimi-K2.6 的备用下载地址分别为 [Kimi-K2.5](https://modelscope.cn/models/moonshotai/Kimi-K2.5/summary) 和 [Kimi-K2.6](https://www.modelscope.cn/models/moonshotai/Kimi-K2.6)。

## 6.2 环境变量设置

MXC5XX/MXC600 配置使用以下通用变量：

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

MXC600U 的 FP8 配置使用以下通用变量：

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

各模型的专用变量在对应部署配置中单独列出；仅执行与所选模型和并行配置对应的变量。多机通信和设备顺序变量按第 2.2 节和第 2.3 节的当前服务器配置设置。

## 6.3 服务部署

### 6.3.1 Kimi-K2.5-FP8（MXC600U，TP8PP4）

**模型准备**：下载 Kimi-K2.5-FP8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
export SGLANG_OPT_USE_COMPRESSOR_V2=False
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_ENABLE_JIT_DEEPGEMM=True
export SGLANG_SHARED_EXPERT_TP1=1
```
**启动服务**：该配置使用 32 张 GPU，执行下方命令启动服务。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --disable-chunked-prefix-cache \
  --mem-fraction-static 0.90 --moe-dense-tp-size 1 \
  --cuda-graph-backend-decode=disabled --max-running-requests 64 \
  --tp-size 8 --pp-size 4 --chunked-prefill-size 2048
```

### 6.3.2 Kimi-K2.5/K2.6-W8A8（MXC5XX/MXC600，TP32DP4）

**模型准备**：下载与目标版本匹配的 Kimi-K2.5 或 Kimi-K2.6 W8A8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。MXC600 平台还需设置 `MACA_DIRECT_DISPATCH=1`；多机通信和设备顺序变量按第 2.2 节和第 2.3 节的当前服务器配置设置。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

仅在 MXC600 平台追加以下变量：

```shell
export MACA_DIRECT_DISPATCH=1
```
**启动服务**：该配置使用 32 张 GPU，按下方命令在四台主机启动服务。

该配置使用 32 张 GPU。在四台主机上执行以下命令，并将 `--node-rank` 分别设置为 `0`、`1`、`2`、`3`。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --disable-chunked-prefix-cache \
  --attention-backend flashinfer \
  --tp-size 32 --dp-size 4 --enable-dp-attention --enable-dp-lm-head \
  --mem-fraction-static 0.86 --quantization w8a8_int8 \
  --chunked-prefill-size 4096 \
  --dist-init-addr <MASTER_IP>:5000 \
  --nnodes 4 \
  --node-rank <0-3> \
  --port 30000
```

---

# 7. Qwen 系列推理部署

本章按“模型准备 → 环境变量设置 → 服务部署 → 第 9 章验证”的顺序覆盖 Qwen3.5-397B 与 Qwen3.6 系列。Qwen3.5-397B-FP8 使用 MXC600U；Qwen3.5-397B-W8A8 和 Qwen3.6 配置使用 MXC5XX 或 MXC600。

## 7.1 模型准备

下载与目标精度一致的 Qwen 模型包，并确认模型目录完整：Qwen3.5-397B-FP8 使用 [Qwen3.5-397B-A17B](https://www.modelscope.cn/models/Qwen/Qwen3.5-397B-A17B)，Qwen3.5-397B-W8A8 使用 [MetaX：Qwen3.5-397B-A17B-W8A8](https://www.modelscope.cn/models/metax-tech/Qwen3.5-397B-A17B-W8A8)，Qwen3.6 使用 [Qwen3.6-27B](https://www.modelscope.cn/models/Qwen/Qwen3.6-27B) 或 [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)。

## 7.2 环境变量设置

MXC5XX/MXC600 配置使用以下通用变量：

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

MXC600U 的 FP8 配置使用以下通用变量：

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

各模型的专用变量在对应部署配置中单独列出；仅执行与所选模型和并行配置对应的变量。

## 7.3 服务部署

### 7.3.1 Qwen3.5-397B-FP8（MXC600U，TP8DP8）

**模型准备**：下载 Qwen3.5-397B-FP8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```
**启动服务**：该配置使用 8 张 GPU，执行下方命令启动服务。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --disable-chunked-prefix-cache \
  --attention-backend flashinfer --tp-size 8 --dp-size 8 \
  --enable-dp-attention --mem-fraction-static 0.85 \
  --mamba-ssm-dtype bfloat16 --disable-overlap-schedule
```

### 7.3.2 Qwen3.5-397B-W8A8（MXC5XX/MXC600，TP16）

**模型准备**：下载 Qwen3.5-397B-A17B-W8A8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MX_ENABLE_DEQUANT_BF16=true
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export MACA_EXT_DIRECT_DISPATCH=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
```
**启动服务**：该配置使用 16 张 GPU，按下方命令在两台主机启动服务。

该配置使用 16 张 GPU。在两台主机上执行以下命令，并分别设置 `--node-rank 0` 和 `--node-rank 1`。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --cuda-graph-max-bs-decode 64 --max-running-requests 64 \
  --chunked-prefill-size 4096 --disable-radix-cache \
  --disable-chunked-prefix-cache --attention-backend flashinfer \
  --tp-size 16 --dp-size 4 --enable-dp-attention --enable-dp-lm-head \
  --mem-fraction-static 0.89 --quantization w8a8_int8 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --mamba-ssm-dtype bfloat16 \
  --dist-init-addr <MASTER_IP>:5000 \
  --nnodes 2 \
  --node-rank <0-1> \
  --port 30000
```

### 7.3.3 Qwen3.6-27B（MXC5XX/MXC600，TP2）

**模型准备**：下载 Qwen3.6-27B BF16 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```
**启动服务**：该配置使用 2 张 GPU，执行下方命令启动服务。

该配置使用 2 张 GPU，在单机执行以下命令。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --disable-chunked-prefix-cache \
  --cuda-graph-max-bs-decode 128 --attention-backend flashinfer \
  --tp-size 2 --mem-fraction-static 0.92
```

### 7.3.4 Qwen3.6-35B-A3B（MXC5XX/MXC600，TP2）

**模型准备**：下载 Qwen3.6-35B-A3B BF16 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```
**启动服务**：该配置使用 2 张 GPU，执行下方命令启动服务。

该配置使用 2 张 GPU，在单机执行以下命令。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --disable-chunked-prefix-cache \
  --cuda-graph-max-bs-decode 128 --attention-backend flashinfer --tp-size 2
```

---

# 8. MiMo 系列推理部署

本章按“模型准备 → 环境变量设置 → 服务部署 → 第 9 章验证”的顺序提供 MiMo-V2-Flash-W8A8 在 MXC5XX 或 MXC600 平台上的部署配置。

## 8.1 模型准备

下载 [MiMo-V2-Flash-W8A8](https://modelscope.cn/models/metax-tech/MiMo-V2-Flash-W8A8) 模型包，并确认模型目录为 W8A8 权重。

## 8.2 环境变量设置

MXC5XX/MXC600 配置使用以下通用变量：

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

各模型的专用变量在对应部署配置中单独列出；仅执行与所选模型和并行配置对应的变量。

## 8.3 服务部署

### 8.3.1 MiMo-V2-Flash-W8A8（MXC5XX/MXC600，TP16DP4）

**模型准备**：下载 MiMo-V2-Flash-W8A8 模型包。  
**环境变量设置**：在启动服务前执行以下完整环境变量。  

```shell
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```
**启动服务**：该配置使用 16 张 GPU，按下方命令在两台主机启动服务。

该配置使用 16 张 GPU。在两台主机上执行以下命令，并分别设置 `--node-rank 0` 和 `--node-rank 1`。

```shell
python3 -m sglang.launch_server --trust-remote-code --cuda-graph-backend-prefill=disabled \
  --model-path <模型目录> \
  --disable-radix-cache --disable-chunked-prefix-cache \
  --attention-backend fa3 --tp-size 16 --dp-size 4 \
  --enable-dp-attention --enable-dp-lm-head \
  --mem-fraction-static 0.86 --quantization w8a8_int8 \
  --dist-init-addr <MASTER_IP>:5000 \
  --nnodes 2 \
  --node-rank <0-1> \
  --port 30000
```

---

# 9. 推理测试

模型服务启动后，按照基本功能测试、随机性能测试和精度测试的顺序执行验证。性能和精度测试使用与部署命令相同的模型、平台和并行配置。

## 9.1 基本功能测试

SGLang 服务启动后，在可访问服务端口的终端执行以下请求：

```python
import requests

url = "http://localhost:30000/generate"
data = {"text": "What is the capital of France?"}

response = requests.post(url, json=data)
print(response.json())
```

远程服务使用以下地址：

```python
url = "http://<SERVER_IP>:30000/generate"
```

服务返回生成结果后，基本服务链路验证通过。模型验收仍须执行第 10.2 至第 10.4 节规定的性能或精度测试。

## 9.2 随机性能测试

随机性能测试使用 `sglang.benchmark.serving` 模块。该模块从 ShareGPT 数据中采样并截断或重复文本，构造指定输入/输出长度的随机请求并发送到已启动的 SGLang 服务；测试结束后在终端输出时延、吞吐等性能指标，并将明细写入 JSONL 文件。

`dataset_path` 指向 ShareGPT 数据集文件 `ShareGPT_V3_unfiltered_cleaned_split.json`。数据集地址为 [anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)。示例：`dataset_path=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json`。该路径不是有效 JSON 文件时，脚本自动下载并缓存该文件。

```shell
# dataset_path：ShareGPT JSON 数据集文件路径
# input_len：每个请求的目标输入 Token 长度
# output_len：每个请求的目标输出 Token 长度
# num_prompts：总请求数
# max_concurrency：最大并发请求数

python3 -m sglang.benchmark.serving \
  --backend sglang \
  --dataset-name random \
  --random-range-ratio 1.0 \
  --dataset-path $dataset_path \
  --random-input-len $input_len \
  --random-output-len $output_len \
  --num-prompts $num_prompts \
  --max-concurrency $max_concurrency \
  --host <SGLang 服务 IP> \
  --port 30000 \
  --output-file random_benchmark.jsonl
```

`--random-range-ratio 1.0` 在目标长度范围内随机采样请求长度；固定输入和输出长度时设置为 `0`。`--host` 和 `--port` 指向被测 SGLang 服务；`--output-file` 指定测试明细文件。服务无法承载当前并发时，降低 `--max-concurrency` 后重新执行测试。

使用以下性能测试组合：

```text
Input/Output: 1024/1024, 3072/1024, 7168/1024, 15360/1024
Concurrency : 1, 16, 32, 64
```

发布测试规范另有规定时，以对应模型的验收配置为准。

## 9.3 C-Eval

C-Eval 使用仓库内的 `benchmark/ceval/bench_sglang.py`。该脚本读取 C-Eval 各学科的 dev/val 数据，构造 few-shot 选择题请求并调用已启动的 SGLang 服务，最后在终端输出准确率、时延和输出吞吐，并将汇总结果写入 JSONL 文件。

确认 C-Eval 数据已预置，将 `<C-Eval 数据目录>` 替换为实际数据目录，并启动 SGLang 服务后执行：

```shell
cd benchmark/ceval

python3 bench_sglang.py \
  --data-path <C-Eval 数据目录> \
  --host <SGLang 服务 IP> \
  --port 30000 \
  --parallel 64 \
  --result-file ceval_result.jsonl
```

`--data-path` 指向 C-Eval 数据目录；`--host` 和 `--port` 指向 SGLang 服务；`--parallel` 指定并发请求数。服务无法承载当前并发时，降低 `--parallel` 后重新执行测试。

## 9.4 MMLU

MMLU 使用仓库内的 `benchmark/mmlu/bench_sglang.py`。该脚本读取 MMLU 的 dev/test CSV 数据，构造 few-shot 选择题请求并调用 SGLang 服务。`--data_dir` 不包含数据时，脚本自动下载 MMLU 数据集。测试结束后，终端输出各学科准确率和平均准确率，并将汇总结果写入 JSONL 文件。

```shell
export TIKTOKEN_CACHE_DIR=<tokenizer缓存目录>

cd benchmark/mmlu

python3 bench_sglang.py \
  --data_dir <MMLU 数据目录> \
  --nsub 60 \
  --host <SGLang 服务 IP> \
  --port 30000 \
  --parallel 64 \
  --result-file mmlu_result.jsonl
```

`--nsub 60` 表示运行全部 60 个学科；`--data_dir` 指定已有数据目录或自动下载目录。服务无法承载当前并发时，降低 `--parallel` 后重新执行测试。

## 9.5 测试结果记录

每次测试必须记录：

| 类别 | 记录内容 |
| --- | --- |
| 环境 | Image TAG/ID、Driver、MACA、Python、Torch、SGLang |
| 硬件 | GPU 型号、卡数、节点数、拓扑 |
| 模型 | 模型名、精度、模型路径/版本 |
| 并行 | TP / DP / PP / CP / MTP 等 |
| 显存 | `mem-fraction-static`、启动后剩余显存、峰值显存 |
| Graph | `cuda-graph-max-bs-decode`、CUDA Graph 后端配置 |
| Prefill | `chunked-prefill-size` 等 |
| 性能 | TTFT、ITL、吞吐、请求吞吐等 |
| 精度 | C-Eval/MMLU 结果 |
| 结论 | Pass / Fail、异常日志位置 |

---
