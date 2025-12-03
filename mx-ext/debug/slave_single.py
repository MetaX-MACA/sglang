import os
import subprocess
import psutil
import signal
import zmq
import threading
import sys
import argparse
import dataclasses
import time
import socket
import re
from typing import Optional, List

os.environ[ 'PYTHONUNBUFFERED' ]='1'

@dataclasses.dataclass
class ProcStatus:
    cmd: Optional[str] = ''
    handle: Optional[None] = None
    output: Optional[List[str]] = None
    store_output: Optional[bool] = False
    ready_flag: Optional[List[str]] = None
    is_ready: Optional[bool] = False


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
                custom_env.pop(env_pair[0], None)
                print(f'## UNSET_ENV: {env_pair[0]}')
            elif len(env_pair) == 2:
                custom_env[env_pair[0]] = replace_env_vars(env_pair[1], custom_env)
                print(f'## SET_ENV: {env_pair[0]}={custom_env[env_pair[0]]}')
            else:
                print(f'## INVALID ENV: {single_env}')
        cmd = cmd[:env_begin]
    print(f'## Run {cmd}')
    proc.cmd = cmd
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
        print(line.strip())

        # check process ready
        if not proc.is_ready and proc.ready_flag:
            for flag in proc.ready_flag:
                if flag in line:
                    proc.is_ready = True
                    break


def run_slave_launch_server(cmd):
    kill_old_process_proc = ProcStatus()
    run_sys_cmd("pkill -9 -f 'sglang::scheduler'", kill_old_process_proc)

    proc = ProcStatus(is_ready=True)
    thread = threading.Thread(target=run_sys_cmd, args=(cmd, proc, ))
    thread.start()
    while True:
        if proc.handle:
            break
        time.sleep(1)
    return proc

def get_gpu_mem_used() -> str:
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


def get_ip() -> str:
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


def get_interface_by_ip(ip) -> str:
    addrs = psutil.net_if_addrs()
    for interface, addresses in addrs.items():
        for addr in addresses:
            if addr.address == ip:
                return interface
    return ''


parser = argparse.ArgumentParser()
parser.add_argument(
    "--current-port",
    type=int,
    default=9999,
    help="client port bind to recv msg"
)
raw_args = parser.parse_args(sys.argv[1:])

ip_addr = get_ip()
if ip_addr == '0.0.0.0':
    print("## unable to get local ip !")
    exit(1)
net_interface = get_interface_by_ip(ip_addr)
if net_interface == '':
    print(f"## unable to get local interface for {ip_addr}!")
    exit(1)

context = zmq.Context()
zmq_socket = context.socket(zmq.REP)
listen_info = f"tcp://{ip_addr}:{raw_args.current_port}"
zmq_socket.bind(listen_info)
print(f'bind to {listen_info}, start recving...')
g_proc_map = {}

while True:
    message = zmq_socket.recv_string()
    print(f'## Recv: ' + message)
    index_flag = message.find('@')
    if index_flag == -1:
        zmq_socket.send_string("invalid message! support GET@xxx or RUN@xxxx or STOP@xxxx")
        continue
    msg_flag = message[:index_flag]
    msg_content = message[index_flag + 1:]
    if msg_flag == 'GET':
        zmq_socket.send_string(f'{net_interface}@{get_gpu_mem_used()}')
    elif msg_flag == 'STOP':
        if msg_content in g_proc_map.keys():
            print(f'kill {g_proc_map[msg_content]}')
            kill_process_all(g_proc_map[msg_content])
            del g_proc_map[msg_content]
            zmq_socket.send_string(f"stop [{msg_content}] success")
        else:
            zmq_socket.send_string(f'[{msg_content}] proc not exist!')
    elif msg_flag =='RUN':
        proc = run_slave_launch_server(msg_content)
        g_proc_map[proc.cmd] = proc.handle
        zmq_socket.send_string(f'run [{proc.cmd}] success')
    else:
        zmq_socket.send_string("invalid message! support RUN@xxxx or STOP@xxxx")

