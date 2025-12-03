import os
import subprocess
import psutil
import signal
import zmq
from datetime import datetime
import threading
import sys
import argparse
import dataclasses
import time
import logging
from utils import configure_logger,kill_all
from typing import Optional, List

logger = logging.getLogger(__name__)

os.environ["PYTHONUNBUFFERED"] = "1"

g_env = os.environ.copy()
LOG_FILE = None

def exit_slave():
    current_pid = os.getpid()
    kill_list = ['sglang']
    kill_all(kill_list)

@dataclasses.dataclass
class ProcStatus:
    handle: Optional[None] = None
    output: Optional[str] = ''
    store_output: Optional[bool] = False
    ready_flag: Optional[List[str]] = None
    is_ready: Optional[bool] = False

def kill_process_all(process):
    exit_slave()
    try:
        cur_process = psutil.Process(process.pid)
        child_pid = cur_process.children(recursive=True)
        for child in child_pid:
            os.kill(child.pid, signal.SIGTERM)
    except Exception as e:
        print(f"kill child process exception {e}")
        pass
    try:
        process.terminate()
    except Exception as e:
        print(f"kill child process exception {e}")
        pass
    # subprocess.run(["pkill", "-9", "-f", "sglang"], check=False)
    status = ProcStatus()
    run_sys_cmd("pkill -9 sglang",status)
    exit_slave()

def run_sys_cmd(cmd: str, proc: ProcStatus):
    """Run |cmd| and return its output."""
    global g_env
    proc.handle = subprocess.Popen(cmd, shell=True, bufsize=1, text=True, encoding='utf-8',
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,env=g_env)
    while True:
        line = proc.handle.stdout.readline()
        if not line and proc.handle.poll() is not None:
            print(f"[{cmd}] exit")
            break

        # store output to memory or not
        if proc.store_output:
            proc.output += line
            # with open(LOG_FILE, 'a') as f:
            #     print(line, file=f)
            logger.info(f'run sys cmd log : {line}')


        print(line.strip())

        # check process ready
        if not proc.is_ready and proc.ready_flag:
            for flag in proc.ready_flag:
                if flag in line:
                    proc.is_ready = True
                    break

def run_slave_launch_server(cmd):
    proc = ProcStatus(store_output=True)
    thread = threading.Thread(target=run_sys_cmd, args=(cmd, proc,))
    thread.start()
    time.sleep(2)
    return proc

def set_env(enviroment_variable):
    global g_env
    index_flag = enviroment_variable.find('=')
    key = enviroment_variable[0:index_flag]
    Value = enviroment_variable[index_flag+1:]
    g_env[key] = Value

def reset_env():
    global g_env
    g_env = os.environ.copy()
    status = ProcStatus()
    run_sys_cmd('printenv', status)

def create_log_file(target_directory,result_path,ip):
    global LOG_FILE
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    LOG_FILE = os.path.join(target_directory, result_path, f"LOG_FILE/{ip}/LOG_FILE_{now}.log")
    create_file(LOG_FILE)

def create_file(filename):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            pass
    print(f"File {filename} has been created or already exists.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-ip",
        type=str,
        help="client ip bind to recv msg",
        required=True,
    )
    parser.add_argument(
        "--current-port",
        type=int,
        default=9998,
        help="client port bind to recv msg"
    )
    parser.add_argument("--target-path", type=str, default="/pde_ai/share/sgl_automation/", help="Target work path")
    parser.add_argument("--result-path", type=str, default="benchmark/benchmark0501/result_2_2/", help="Path for storing results")
    raw_args = parser.parse_args(sys.argv[1:])

    create_log_file(raw_args.target_path,raw_args.result_path,raw_args.current_ip)

    configure_logger(log_file = LOG_FILE)
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        listen_info = f"tcp://{raw_args.current_ip}:{raw_args.current_port}"
        socket.bind(listen_info)
        print(f'bind to {listen_info}, start recving...')
        g_proc_map = {}

        while True:
            message = socket.recv_string()
            # with open(LOG_FILE, 'a') as f:
            #     print(message, file=f)
            logger.info(f'## Recv: {message}')
            
            index_flag = message.find('@')
            if index_flag == -1:
                socket.send_string("invalid message! support RUN@xxxx or STOP@xxxx")
                continue
            msg_flag = message[:index_flag]
            msg_content = message[index_flag + 1:]
            if msg_flag == 'STOP':
                if msg_content in g_proc_map.keys():
                    # print(f'kill {g_proc_map[msg_content]}')
                    logger.info(f'kill {g_proc_map[msg_content]}')
                    kill_process_all(g_proc_map[msg_content])
                    socket.send_string(f"stop [{msg_content}] success")
                else:
                    socket.send_string(f'[{msg_content}] proc not exist!')
            elif msg_flag == 'RUN':
                proc = run_slave_launch_server(msg_content)
                g_proc_map[msg_content] = proc.handle
                socket.send_string(f"run [{msg_content}] success")
            elif msg_flag == "ENVSET":
                set_env(msg_content)
                socket.send_string(f"envset [{msg_content}] success")
            elif msg_flag == "ENVRESET":
                reset_env()
                socket.send_string(f"envreset [{msg_content}] success")
            elif msg_flag == "EXIT":
                socket.send_string(f"EXIT [{msg_content}] success")
                sys.exit(0)
            else:
                logger.info(f'{msg_flag} invalid message! support RUN@xxxx or STOP@xxxx or ENVSET@ or EXIT')
                socket.send_string("invalid message! support RUN@xxxx or STOP@xxxx or ENVSET@ or EXIT")
                
    finally:
        exit_slave()
        sys.exit(0)