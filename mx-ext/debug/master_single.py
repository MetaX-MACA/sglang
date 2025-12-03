import os
import subprocess
import psutil
import signal
import zmq
import threading
import sys
import socket
import dataclasses
import time
import re
import colorama
from typing import Optional, List

os.environ[ 'PYTHONUNBUFFERED' ]='1'


@dataclasses.dataclass
class ProcStatus:
    handle: Optional[None] = None
    output: Optional[List[str]] = None
    store_output: Optional[bool] = False
    print_output: Optional[bool] = True
    ready_flag: Optional[List[str]] = None
    is_ready: Optional[bool] = False


def print_error(msg):
    print(f'{colorama.Fore.RED}{msg}{colorama.Style.RESET_ALL}')


def kill_process_all(process):
    try:
        itself = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    try:
        itself.kill()

        # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
        # so we send an additional signal to kill them.
        itself.send_signal(signal.SIGQUIT)
    except psutil.NoSuchProcess:
        pass


def replace_env_vars(env_value, custom_env):
    def replacer(match):
        # 处理 ${VAR} 形式
        if match.group(1):
            var_name = match.group(1)
        # 处理 $VAR 形式
        else:
            var_name = match.group(2)
        return custom_env.get(var_name, match.group(0))  # 找不到变量则保持原样
    
    # 正则匹配 ${VAR} 或 $VAR，但不匹配转义的\$
    pattern = r'(?<!\\)\$\{([^}]+)\}|(?<!\\)\$([A-Za-z_][A-Za-z0-9_]*)'
    return re.sub(pattern, replacer, env_value)


def run_sys_cmd(cmd: str, proc: ProcStatus):
    """Run |cmd| and return its output."""
    custom_env = os.environ.copy()
    env_begin = cmd.find('@ENV:')
    if env_begin > 0:
        env_list = cmd[env_begin+5:].split(';')
        for single_env in env_list:
            env_pair = single_env.split('=')
            if len(env_pair) == 1:
                print(f'## UNSET_ENV: {env_pair[0]}')
            elif len(env_pair) == 2:
                custom_env[env_pair[0]] = replace_env_vars(env_pair[1], custom_env)
                print(f'## SET_ENV: {env_pair[0]}={custom_env[env_pair[0]]}')
            else:
                print(f'## INVALID ENV: {single_env}')
        cmd = cmd[:env_begin]
    print(f'## Run {cmd}')
    proc.handle = subprocess.Popen([cmd], shell=True, bufsize=1, text=True, encoding='utf-8',
                                   env=custom_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        line = proc.handle.stdout.readline()
        if not line and proc.handle.poll() is not None:
            print(f"[{cmd}] exit")
            break

        # store output to memory or not
        if proc.store_output:
            if proc.output:
                proc.output.append(line.strip())
            else:
                proc.output = [line.strip()]

        if proc.print_output:
            print(line.strip())

        # check process ready
        if not proc.is_ready and proc.ready_flag:
            for flag in proc.ready_flag:
                if flag in line:
                    proc.is_ready = True
                    break


class ConnectionMgr:
    def __init__(self, slave_ip_list:List, socket_timeout: int = 5) -> None:
        self.local_ip = None
        self.local_net = None
        self.local_proc_map = {}
        self.slave_ip_list = slave_ip_list
        self.slave_nets = []
        self.slave_sockets = []
        self.context = zmq.Context()
        self.socket_timeout = socket_timeout

    def connect(self) -> None:
        self.local_ip = self._get_local_ip()
        if self.local_ip == '0.0.0.0':
            print_error("## unable to get local ip !")
            exit(1)
        self.local_net = self._get_interface_by_ip(self.local_ip)
        if self.local_net == '':
            print_error(f"## unable to get local interface for {self.local_ip}!")
            exit(1)
        mem_used_info = self._get_gpu_mem_used()
        mem_check_status = self._check_gpu_in_use(mem_used_info)
        if mem_check_status != '':
            print_error(f"local {mem_check_status}")
            exit(1)
        
        # connect to all slave
        for slave_ip in self.slave_ip_list:
            socket = self.context.socket(zmq.REQ)
            socket.RCVTIMEO = self.socket_timeout * 1000
            index_port = slave_ip.find(':')
            if index_port > 0:
                socket.connect(f"tcp://{slave_ip}")
            else:
                socket.connect(f"tcp://{slave_ip}:9999")
            socket.send_string(f'GET@INFO')
            try:
                net, slave_mem_used_info = socket.recv_string().split('@')
                print(f"## connect to {slave_ip} successfully!")
                self.slave_nets.append(net)
                slave_mem_check_status = self._check_gpu_in_use(slave_mem_used_info)
                if slave_mem_check_status != '':
                    print_error(f"{slave_ip} {slave_mem_check_status}")
                    exit(1)
            except zmq.Again:
                print_error(f"## connect to {slave_ip} timeout, exit!")
                exit(1)
            self.slave_sockets.append(socket)

    def clear(self) -> None:
        for sock in self.slave_sockets:
            sock.close()

    def run_local_cmd(self, cmd, envs):
        kill_old_process_proc = ProcStatus()
        run_sys_cmd("pkill -9 -f 'sglang::scheduler'", kill_old_process_proc)

        cmd_with_env = cmd
        if envs is not None:
            cmd_with_env = f'{cmd}@ENV:{";".join(envs)}'
        proc = ProcStatus(store_output=True, ready_flag=['Init torch distributed begin.'])
        thread = threading.Thread(target=run_sys_cmd, args=(cmd_with_env, proc, ))
        thread.start ()
        while True:
            if proc.is_ready:
                break
            time.sleep(1)
        self.local_proc_map[cmd] = proc.handle
        return proc

    def stop_local_cmd(self, cmd):
        if cmd in self.local_proc_map.keys():
            print(f'## kill {self.local_proc_map[cmd]}')
            kill_process_all(self.local_proc_map[cmd])
        else:
            print(f'[{cmd}] proc not exist!')

    def run_slave_cmd(self, socket, cmd, envs) -> None:
        if envs is not None:
            socket.send_string(f'RUN@{cmd}@ENV:{";".join(envs)}')
        else:
            socket.send_string(f'RUN@{cmd}')
        message = socket.recv_string()
        print(f'## Recv {message}')

    def stop_slave_cmd(self, socket, cmd) -> None:
        socket.send_string(f'STOP@{cmd}')
        message = socket.recv_string()
        print(f'## Recv {message}')

    def _get_local_ip(self) -> str:
        host_ip = os.getenv("SGLANG_HOST_IP", "") or os.getenv("HOST_IP", "")
        if host_ip:
            return host_ip
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
            return s.getsockname()[0]
        except Exception:
            pass
        return "0.0.0.0"
    
    def _get_interface_by_ip(self, ip) -> str:
        addrs = psutil.net_if_addrs()
        for interface, addresses in addrs.items():
            for addr in addresses:
                if addr.address == ip:
                    return interface
        return ''
    
    def _get_gpu_mem_used(self) -> str:
        mem_use = ProcStatus(store_output=True)
        run_sys_cmd("mx-smi", mem_use)
        mem_pos = 0
        percent_pos = 0
        used_percent = '0'
        mem_list = []
        for line in mem_use.output:
            if 'Bus-id' in line:
                mem_pos = line.find('Bus-id')
                percent_pos = line.find('GPU-Util')
                continue
            if mem_pos == 0 or percent_pos == 0:
                continue

            if 'MetaX' in line:
                percent_end = line.find('%', percent_pos)
                used_percent = line[percent_pos:percent_end]
            elif '65536 MiB' in line:
                mem_end = line.find('/', mem_pos)
                used_mem_mb = line[mem_pos:mem_end].strip()
                mem_list.append(f'{used_mem_mb}_{used_percent}')
            else:
                continue
        return ','.join(mem_list)
    

    def _check_gpu_in_use(self, mem_used) -> str:
        gpu_mem_used = mem_used.split(',')
        for gpu_id, mem_info in enumerate(gpu_mem_used):
            used_mem, used_per = mem_info.split('_')
            if int(used_mem) >= 1000 or int(used_per) > 0:
                return f'GPU {gpu_id} are in used({used_mem}, {used_per}), please check process!!'
        return ''


class Task:
    def __init__(self, connect_mgr: ConnectionMgr, task_cmd: str, test_envs: List = None, test_cmd: List = None) -> None:
        self.connect_mgr = connect_mgr
        self.task_cmd = task_cmd
        self.master_cmd = None
        self.slave_cmds = {}
        self.stopped = False
        self.envs=test_envs

    def run(self, stop_server_after_test: bool = False) -> None:
        self.start_server()
        self.test()
        if stop_server_after_test:
            self.stop_server()

    def start_server(self) -> None:
        nodes_num = len(self.connect_mgr.slave_sockets) + 1
        if nodes_num > 1:
            self.task_cmd += f' --dist-init-addr {self.connect_mgr.local_ip}:5000 --nnodes {nodes_num}'
            self.master_cmd = f'{self.task_cmd} --node-rank 0'
        else:
            self.master_cmd = self.task_cmd

        # start run local cmd
        master_proc = self.connect_mgr.run_local_cmd(self.master_cmd, self.envs[0])

        # start run slave cmd
        for i, sock in enumerate(self.connect_mgr.slave_sockets):
            slave_env = self.envs[0]
            if len(self.envs) > 1:
                slave_env = self.envs[i + 1]
            slave_cmd = f'{self.task_cmd} --node-rank {i + 1}'
            self.connect_mgr.run_slave_cmd(sock, slave_cmd, slave_env)
            self.slave_cmds[sock] = slave_cmd

        # wait 
        while True:
            if self.stopped:
                print('## server has been stopped !')
                return
            if 'The server is fired up and ready to roll' in master_proc.output:
                break
            time.sleep(1)
        print('## start all server successful !')

    def stop_server(self) -> None:
        # kill master
        self.connect_mgr.stop_local_cmd(self.master_cmd)
        # kill slave
        for sock, cmd in self.slave_cmds.items():
            self.connect_mgr.stop_slave_cmd(sock, cmd)
        self.slave_cmds.clear()
        self.stopped = True

    def test(self) -> None:
        # todo
        pass


class TaskScheduler:
    def __init__(self, task_config: str) -> None:
        self.config_file = task_config
        self.slave_ip_list = []
        self.connection_mgr = None
        self.task_list = []
        self.current_task = None

    def run_task(self) -> None:
        self.generate_task()
        if len(self.task_list) == 0:
            print('## no task to run !')
            return
        for i, task in enumerate(self.task_list):
            self.current_task = task
            print(f'######################################################################## TASK LINE ({i + 1}/{len(self.task_list)})########################################################################')
            task.run()

    def stop(self) -> None:
        if self.current_task:
            self.current_task.stop_server()

    def generate_task(self) -> None:
        # todo from config file
        # self.slave_ip_list = ['192.168.3.2', '192.168.3.181', '192.168.3.143']
        # self.slave_ip_list = ['192.168.0.26']#, '192.168.0.169', '192.168.3.97']
        self.slave_ip_list = [
            '10.2.179.92',
            # '10.2.179.93',
            # '10.2.179.94',
            ]
        self.connection_mgr = ConnectionMgr(self.slave_ip_list, socket_timeout=5)
        self.connection_mgr.connect()
        # node rank 和 server 等参数会自动补全，不需要填写，这里只需要填写其他参数即可
        task_cmd = f'python3 -m sglang.launch_server --model-path /mnt/data/DeepSeek-R1-W8A8/vllm_quant_model \
--dp 4 --tp 16 --enable-dp-attention --trust-remote-code --chunked-prefill-size 2048' 
#         task_cmd = f'python3 -m sglang.launch_server --model-path /mnt/data/DeepSeek-R1-W8A8/vllm_quant_model \
# --tp 16 --trust-remote-code '#--chunked-prefill-size 1024 --disable-cuda-graph' 
#         task_cmd = f'python3 -m sglang.launch_server --model-path /mnt/data/DeepSeek-R1-W8A8/vllm_quant_model \
# --tp 32 --trust-remote-code --disable-cuda-graph' 

        common_env = [
            'MACA_PATH=/opt/maca',
            'MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin',
            # 'MXSHMEM_LIB_PATH=${MACA_PATH}/mxshmem/',
            # 'DEVICE_MEM_CACHE_FLAG=0',
            # 'NVSHMEM_DISABLE_CUDA_VMM=1',
            # 'NVSHMEM_SYMMETRIC_SIZE=524288000',
            #'NVSHMEM_BOOTSTRAP_TWO_STAGE=1',',
            # 'NVSHMEM_IB_ENABLE_IBGDA=1',
            # 'LD_LIBRARY_PATH=${MXSHMEM_LIB_PATH}/lib:${MACA_PATH}/mxgdrcopy/lib/:${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH',
            'LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH',
            'CUDA_PATH=$MACA_PATH/tools/cu-bridge',
            'PATH=$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH',
            'SGLANG_IS_FLASHINFER_AVAILABLE=False',
            # 'MACA_SMALL_PAGESIZE_ENABLE=1',
            # 'ENABLE_TORCH_PROFILE=true',
            # 'PROFILE_CUDA_ONLY=true',
            # 'SGLANG_TORCH_PROFILER_DIR=/pde_ai/share/xinyang/profile',
        ]
        local_env = [f'GLOO_SOCKET_IFNAME={self.connection_mgr.local_net}']
        slave_envs = [[f'GLOO_SOCKET_IFNAME={slave_net}'] for slave_net in self.connection_mgr.slave_nets]

        task_envs = [
            common_env + local_env, # master
        ] + [common_env + env for env in slave_envs]
        task = Task(self.connection_mgr, task_cmd, test_envs=task_envs)
        self.task_list.append(task)


class SignalHandler:
    def __init__(self, task_scheduler: TaskScheduler):
        self.task_scheduler = task_scheduler
    
    def __call__(self, signum, frame):
        print('')
        print('## Recv stopped signal, exit !')
        self.task_scheduler.stop()
        exit(0)


if __name__ == "__main__":
    # all slave ip 
    task_scheduler = TaskScheduler('')
    handler = SignalHandler(task_scheduler)
    signal.signal(signal.SIGINT, handler)
    task_thread = threading.Thread(target=task_scheduler.run_task)
    task_thread.start()
    task_thread.join()
