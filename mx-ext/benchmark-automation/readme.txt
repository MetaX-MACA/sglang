master节点启动命令：
    python3 -m master_launch --target-path xxx --json-file benchmark.json 
    --target-path 默认是 /workspace/benchmark/，json-file要去这个路径下面找，生成的benchmark result,保存的log等都在这个路径
slave节点启动命令：
    python3 -m slave_launch  --current-ip 192.168.2.13
    python3 -m slave_launch  --current-ip 192.168.2.14
    python3 -m slave_launch  --current-ip 192.168.2.16
    current-ip当前机器的ip