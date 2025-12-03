import subprocess
import os
import signal
from datetime import datetime
import threading
import csv
import re
import json
from itertools import product
import time
import sys
import psutil
import signal
import zmq
import argparse
import pandas as pd
import dataclasses
import logging
from utils import configure_logger,kill_all
from typing import Optional, List

logger = logging.getLogger(__name__)


os.environ["PYTHONUNBUFFERED"] = "1"
SERVER_READY = False
TOTAL_NODES_NUM = 4
RATE_OF_PROGRESS = 0
LOG_FILE = None
REAL_PROGRESS_FILE = None
GET_RESULT = False
TIMEOUT_DURATION = 60*2.678
CHECK_TIMEOUT_ON = False
CHECK_OTHER_ABNORMAL_ON = False
PROFILE_TEST_BASH = "/pde_ai/share/sgl_automation/profile/run_test_bash.sh"
# ########################################################################### Program Logic Control ###########################################################################
g_env = os.environ.copy()

@dataclasses.dataclass
class ProcStatus:
    handle: Optional[None] = None
    output: Optional[str] = ''
    store_output: Optional[bool] = False
    print_output: Optional[bool] = True
    ready_flag: Optional[List[str]] = None
    is_ready: Optional[bool] = False
    statu: Optional[int] = 0
    is_master: Optional[bool] = False
    last_update_time: Optional[float] = time.time()

g_master_proc = ProcStatus()
g_bench_proc = ProcStatus()
g_proc_map = {}

G_MASTER_LOCK = threading.Lock()


def master_launch_server_abnormal():
    global g_master_proc
    with G_MASTER_LOCK:
        return g_master_proc.statu != 0

def set_g_master_statu(statu):
    global g_master_proc
    with G_MASTER_LOCK:
        g_master_proc.statu = statu

def abnormal_condition(bench_statu):
    if master_launch_server_abnormal() or bench_statu != 0:
        return True
    else:
        return False

def kill_process_all(process):
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


def exit_master():
    kill_list = ['sglang','python','python3']
    kill_all(kill_list)


def stop_all(task, slave_cmd_map):
    global LOG_FILE
    exit_master()
    try:
        stop_master_bench_client()
        for sock, cmd in slave_cmd_map.items():
            stop_slave_launch_server(sock, cmd)
        stop_master_launch_server(node_rank_command(task['launch_server'], 0))
        # subprocess.run(["pkill", "-9", "-f", "sglang"], check=False)
        status = ProcStatus()
        run_sys_cmd("pkill -9  sglang",status)
        exit_master()
    except Exception as e:
        # subprocess.run(["pkill", "-9", "-f", "sglang"], check=False)
        status = ProcStatus()
        run_sys_cmd("pkill -9  sglang",status)
        exit_master()

        logger.error(f"stop_all error:{e}")


def run_sys_cmd(cmd: str, proc: ProcStatus):
    """Run [cmd] and return its output."""
    global g_env
    logger.info(cmd)
    
    proc.handle = subprocess.Popen(cmd, shell=True, bufsize=1, text=True, encoding='utf-8',
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,env=g_env)
    while True:
        line = proc.handle.stdout.readline()
        if not line and proc.handle.poll() is not None:
            print(f"[{cmd}] exit")
            with G_MASTER_LOCK:
                proc.statu = proc.handle.poll()
                return proc.statu

        proc.last_update_time = time.time()

        # store output to memory or not
        if proc.store_output:
            proc.output += line

            logger.info(f'run sys cmd log : {line}')

        if proc.print_output:
            print(line.strip())

        # check process ready
        if not proc.is_ready and proc.ready_flag:
            for flag in proc.ready_flag:
                if flag in line:
                    proc.is_ready = True
                    break
    return 0

def run_bench_cmd(cmd: str, proc: ProcStatus):
    """Run [cmd] and return its output."""

    logger.info(cmd)
    global g_env
    global g_master_proc
    proc.handle = subprocess.Popen(cmd, shell=True, bufsize=1, text=True, encoding='utf-8',
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,env=g_env)
    while True:
        line = proc.handle.stdout.readline()
        if not line and proc.handle.poll() is not None:
            print(f"[{cmd}] exit")
            proc.statu = proc.handle.poll()
            return proc.statu

        # store output to memory or not
        if proc.store_output:
            proc.output += line

            logger.info(line)

        if proc.print_output:
            print(line.strip())

        if master_launch_server_abnormal():
            try:
                proc.handle.terminate()
            except Exception as e:
                print(f"kill child process exception {e}")
            pass
            return g_master_proc.statu
    return 0

def run_master_launch_server(target_directory, cmd):
    global g_master_proc, g_proc_map
    os.chdir(target_directory)
    with G_MASTER_LOCK:
        g_master_proc = ProcStatus(store_output=True, print_output=True, ready_flag=["The server is fired up and ready to roll!"],
                                    is_master=True, last_update_time=time.time())
    thread = threading.Thread(target=run_sys_cmd, args=(cmd, g_master_proc,))
    thread.start()
    time.sleep(2)
    g_proc_map[cmd] = g_master_proc.handle
    return g_master_proc

def stop_master_launch_server(cmd):
    with G_MASTER_LOCK:
        global g_proc_map
        if cmd in g_proc_map.keys():
            print(f'kill {g_proc_map[cmd]}')
            kill_process_all(g_proc_map[cmd])
        else:
            print(f'{cmd} proc not exist!')

def run_master_bench_client(cmd):
    global g_bench_proc
    g_bench_proc = ProcStatus(store_output=True, print_output=False)
    statu = run_bench_cmd(cmd, g_bench_proc)
    print(g_bench_proc.output)
    return statu, g_bench_proc.output

def stop_master_bench_client():
    global g_bench_proc
    print(f'kill {g_bench_proc.handle}')
    kill_process_all(g_bench_proc.handle)


def run_child_launch_server(launch_server_command, machine_list):
    slave_cmd_map = {}
    for rank in range(len(machine_list)):
        if 0 == rank:
            continue
        if rank >= (TOTAL_NODES_NUM):
            break
        command = node_rank_command(launch_server_command,rank)
        print(f"Command: {command}")
        slave_cmd_map[machine_list[rank]] = command
        run_slave_launch_server(machine_list[rank], command)
    return slave_cmd_map

def run_slave_launch_server(socket, cmd):
    socket.send_string(f'RUN@{cmd}')
    message = socket.recv_string()
    print(f'## Recv {message}')

def set_slave_env(socket, cmd):
    socket.send_string(f'ENVSET@{cmd}')
    message = socket.recv_string()
    print(f'## Recv {message}')

def run_slave_cmd(machine_list, cmd):
    for rank in range(len(machine_list)):
        if 0 == rank:
            continue
        if rank >= (TOTAL_NODES_NUM):
            break
        machine_list[rank].send_string(cmd)
        message = machine_list[rank].recv_string()
        print(f'## Recv {message}')

def reset_slave_env(machine_list):
    run_slave_cmd(machine_list,f'ENVRESET@ENVRESET')

def exit_slave(machine_list):
    run_slave_cmd(machine_list,f'EXIT@EXIT')

def stop_slave_launch_server(socket, cmd):
    socket.send_string(f'STOP@{cmd}')
    message = socket.recv_string()
    print(f'## Recv {message}')

def wait_server_ready():
    global g_master_proc
    while True:
        time.sleep(5)
        if g_master_proc.is_ready:
            return True
            pass
        if master_launch_server_abnormal():
            return False

def store_bench_result(result_file, result, result_flag, task_name, command,f,error=None):
    if result_flag == 'Exception':
        print(f"[{task_name}_{command} error : {error}]", file=f)
        print(f"        test_result:  Exception", file=f)
        g_result_status.test_result = 'Exception'
    elif result_flag == 'fail':
        print(f"[{task_name}_{command} error ]")
        print(f"        test_result:  fail", file=f)
        g_result_status.test_result = 'fail'
    elif result_flag == 'pass':
        print(f"        test_result:  pass", file=f)
        g_result_status.test_result = 'pass'
    with open(result_file, "w") as result_file:
        result_file.write(f"Command: {command}\n")
        result_file.write(f"{result}\n")


def run_bench_serving(target_path, init_result_path, task, init_result_file=False):
    global GET_RESULT, g_master_proc, g_result_status
    benchmark_list = task['bench_serving'][g_result_status.bench_serving_id:]
    task_name = task['task_name']
    os.chdir(target_path)
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    result_path = os.path.join(init_result_path, "bench_result",f"{task_name}")
    if  task["task_type"] == "acc_ceval":
        result_path = os.path.join(init_result_path, "acc_result/ceval",f"{task_name}")
    elif task["task_type"] == "acc_mmlu":
        result_path = os.path.join(init_result_path, "acc_result/mmlu",f"{task_name}")
    if GET_RESULT:
        get_result(result_path)
        return None, False

    statu = 0
    start_id = g_result_status.bench_serving_id
    if master_launch_server_abnormal():
        return -1, False
    for command_id, command in enumerate(benchmark_list):
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # now = datetime.now().strftime("%Y_%m")
        bench_serving_args_str = get_bench_serving_args_str(command)


        result_file_txt = os.path.join(result_path, f"result.txt")
        result_file_jsonl = os.path.join(result_path, f"result.jsonl")

        if task["task_type"] == "benchmark" or task["task_type"] == "profile":
            result_file_txt = os.path.join(result_path, f"{bench_serving_args_str}_result.txt")
            result_file_jsonl = os.path.join(result_path, f"{bench_serving_args_str}_result.jsonl")
            command = f"{command} --output-file {result_file_jsonl}"
            create_file(result_file_jsonl)

        elif task["task_type"] == "acc_ceval":
            result_file_txt = os.path.join(result_path, f"acc_ceval_result.txt")
            command = f"{command} --save_dir {result_path}"

        elif task["task_type"] == "acc_mmlu":
            result_file_txt = os.path.join(result_path, f"acc_mmlu_result.txt")
            command = f"{command} --save_dir {result_path}"
            result_file_jsonl = os.path.join(result_path, f"acc_mmlu_result.jsonl")
            command = f"{command} --result-file {result_file_jsonl}"
            create_file(result_file_jsonl)

        create_file(result_file_txt)

        if init_result_file:
            continue

        with open(os.path.join(target_path, REAL_PROGRESS_FILE), 'a') as f:
            try:
                g_result_status.bench_serving_id = start_id + command_id
                print(f"    Bench_serving_id: {g_result_status.bench_serving_id}", file=f)
                print(f"        Bench_Serving: {command}", file=f)
                statu, result = run_master_bench_client(command)
            except Exception as e:
                store_bench_result(result_file_txt,' ','Exception',task['task_name'],command,f,e)
            else:
                if statu == 0:
                    store_bench_result(result_file_txt,result,'pass',task['task_name'],command,f,error=None)
                else:
                    store_bench_result(result_file_txt,result,'fail',task['task_name'],command,f,error=None)
        if master_launch_server_abnormal():
            break
    get_result(task['task_server_args'],result_path)
    return statu, True

def set_env(environment_variables, machine_list):
    global g_env
    g_env = os.environ.copy()
    reset_slave_env(machine_list)
    for key, value in environment_variables.items():
        env_variable = f"{key}={value}"
        for rank in range(len(machine_list)):
            if 0 == rank:
                g_env[key] = value
                continue
            if rank >= (TOTAL_NODES_NUM):
                break
            set_slave_env(machine_list[rank],env_variable)

    logger.info(f"environment_variables:\n")
    status = ProcStatus(store_output=True, print_output=True)
    run_sys_cmd('printenv', status)
    print("*****************set env end*************************")

def set_profile_path(task_name, environment_variables):
    init_path = environment_variables["SGLANG_TORCH_PROFILER_DIR"]
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    profile_path = os.path.join(init_path, f"profile_{task_name}_{now}")
    environment_variables["SGLANG_TORCH_PROFILER_DIR"] = profile_path
    if profile_path and not os.path.exists(profile_path):
        os.makedirs(profile_path)
    return profile_path

def init_profile_bash():
    with open(PROFILE_TEST_BASH,'w') as f:
        print(f"#!/bin/bash\n",file=f)
        print(f"cd /pde_ai/share/sgl_automation/profile\n",file=f)

def set_profile_bash(profile_path):
    with open(PROFILE_TEST_BASH,'a') as f:
        print(f'echo "*******start run {profile_path}*******"\n',file=f)
        print(f'bash run-c500-test.sh {profile_path}\n',file=f)
        print(f'echo "*******run {profile_path} end*********"\n',file=f)

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

def wait_profile_finish(environment_variables):
    profile_path = environment_variables["SGLANG_TORCH_PROFILER_DIR"]
    global g_master_proc
    finish_flag = False
    last_size = 0
    while not finish_flag:
        if "Profiling done" in g_master_proc.output:
            profile_size = get_folder_size(profile_path)
            if profile_size == last_size:
                finish_flag = True
            else:
                last_size = profile_size
                time.sleep(3)
        
# ########################################################################### Program Logic Control ###########################################################################

# ########################################################################### simulation_crash ###########################################################################

def simulation_crash(cmd):
    thread = threading.Thread(target=crash_the_server, args=(cmd,))
    thread.start()

def crash_the_server(cmd):
    global g_proc_map
    time.sleep(2)
    if cmd in g_proc_map.keys():
        process = g_proc_map[cmd]
    else:
        print(f"{cmd} proc not exist!")
    try:
        cur_process = psutil.Process(process.pid)
        child_pid = cur_process.children(recursive=True)
        for child in child_pid:
            os.kill(child.pid, signal.SIGTERM)
    except Exception as e:
        print(f"kill child process exception {e}")
    # pass
    # try:
    #     process.terminate()
    # except Exception as e:
    #     print(f"kill child process exception {e}")
    # pass

# ########################################################################### simulation_crash ###########################################################################


# ########################################################################### check_abnormal ###########################################################################    
def check_other_abnormal(task, slave_cmd_map):
    other_abnormal_flags = ["Gracefully exiting... remaining number of requests",
                            "Watchdog timeout (self.watchdog_timeout=300)",
                            "torch.OutOfMemoryError: CUDA out of memory.",
                            "Forcing disable 'CUTLASS' backend as it is not supported in maca platform.",
                            "TypeError: launcher() got an unexpected keyword argument 'scenario'",
                            "Exception: Capture cuda graph failed:"]
    global g_master_proc, CHECK_OTHER_ABNORMAL_ON, TIMEOUT_DURATION
    CHECK_OTHER_ABNORMAL_ON = True
    ABNORMAL_flag = False
    while not ABNORMAL_flag:
        for abnormal_flags in other_abnormal_flags:
            if abnormal_flags in g_master_proc.output:
                ABNORMAL_flag = True
                logger.warning(f"********************************abnormal********************************")
                logger.warning(f"****************************{abnormal_flags}****************************")

                break

        current_time = time.time()
        time_diff = current_time - g_master_proc.last_update_time
        if time_diff >= TIMEOUT_DURATION:
            ABNORMAL_flag = True
            logger.warning(f"********************************abnormal********************************")
            logger.warning(f"****************************timeout****************************")

    set_g_master_statu(-1)
    stop_all(task, slave_cmd_map)
    CHECK_OTHER_ABNORMAL_ON = False

def start_check_other_abnormal(task, slave_cmd_map):
    timeout_thread = threading.Thread(target=check_other_abnormal, args=(task, slave_cmd_map,))
    timeout_thread.daemon = True
    timeout_thread.start()

# ########################################################################### check_abnormal ###########################################################################


# ########################################################################### task_management ###########################################################################
@dataclasses.dataclass
class ResultStatus:
    launch_server_id: Optional[int] = 0
    bench_serving_id: Optional[int] = 0
    test_result: Optional[str] = ''

g_result_status = ResultStatus()

def all_task_end(all_task_list):
    global g_result_status
    if g_result_status.launch_server_id == (len(all_task_list)-1) and \
       g_result_status.bench_serving_id == (len(all_task_list[-1]["bench_serving"])-1):
        return True
    else:
        return False

def this_task_end(task):
    global g_result_status
    if g_result_status.bench_serving_id == (len(task["bench_serving"])-1):
        return True
    else:
        return False

def write_next_task(all_task_list, task, args):
    global g_result_status
    next_task_command = " "
    launch_server_id = g_result_status.launch_server_id
    bench_serving_id = g_result_status.bench_serving_id
    if this_task_end(task):
        if all_task_end(all_task_list):
            launch_server_id = 0
            bench_serving_id = 0
            
        else:
            launch_server_id = g_result_status.launch_server_id + 1
            bench_serving_id = 0
    else:
        bench_serving_id = g_result_status.bench_serving_id + 1

    next_task_command = f" --launchserver-id {launch_server_id} --benchserving-id {bench_serving_id} --current-port {(args.current_port)}"
    with open(args.progress_file, 'w') as f:
        print(next_task_command,file=f)

def update_result_status(all_task_list, task, run_bench_flag,args):
    global g_result_status
    write_next_task(all_task_list,task,args)

    logger.debug(f"RATE_OF_PROGRESS: {g_result_status.launch_server_id+1} launch_server_id: {g_result_status.launch_server_id} bench_serving_id: {g_result_status.bench_serving_id}")
    if not run_bench_flag:
        logger.debug(f"***************launch_server fail******************")
        if g_result_status.launch_server_id == (len(all_task_list)-1):
            return "all finishi"
        else:
            g_result_status.launch_server_id += 1
            g_result_status.bench_serving_id = 0
            return "this fail"

    logger.debug(f"***************test finish******************")

    if this_task_end(task):
        if all_task_end(all_task_list):
            return "all finishi"
        else:
            g_result_status.launch_server_id += 1
            g_result_status.bench_serving_id = 0
            return "this finishi"
    else:
        g_result_status.bench_serving_id += 1
    return "this continue"

# ########################################################################### task_management ###########################################################################



# ########################################################################### data processing ###########################################################################
def parse_json(json_file_path,args):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    task_list = generate_commands(data,args)
    machine_list = generate_machine(data,args.current_port)
    for task in task_list:
        print(f"Task: {task['task_name']}")
        print("Launch Server Commands:")
        print(f" {task['launch_server']}")
        print("Benchmark Commands:")
        for cmd in task['bench_serving']:
            print(f" {cmd}")
    return task_list, machine_list

def create_log_file(target_directory,result_path):
    global LOG_FILE
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    LOG_FILE = os.path.join(target_directory, result_path, f"LOG_FILE/MASTER/LOG_FILE_{now}.log")
    create_file(LOG_FILE)

def create_real_progress_file(target_directory,result_path):
    global REAL_PROGRESS_FILE
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    REAL_PROGRESS_FILE = os.path.join(target_directory,result_path, f"REAL_PROGRESS_FILE/REAL_PROGRESS_FILE_{now}.txt")
    create_file(REAL_PROGRESS_FILE)
    with open(REAL_PROGRESS_FILE, 'w') as f:
        pass

def create_file(filename):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            pass
    print(f"File {filename} has been created or already exists.")

def get_bench_serving_args(command):
    input_len_match = re.search(r"--random-input-len\s+(\d+)", command)
    output_len_match = re.search(r"--random-output-len\s+(\d+)", command)
    num_prompt_match = re.search(r"--num-prompt\s+(\d+)", command)

    input_len = input_len_match.group(1) if input_len_match else "0"
    output_len = output_len_match.group(1) if output_len_match else "0"
    num_prompt = num_prompt_match.group(1) if num_prompt_match else "0"

    return input_len,output_len,num_prompt

def get_bench_serving_args_str(command):
    input_len,output_len,num_prompt = get_bench_serving_args(command)
    bench_args_str = f"In{input_len}-out{output_len}-bs{num_prompt}"
    print(bench_args_str)
    return bench_args_str


def get_launch_server_args(task_server_id, command):
    server_args = {
                    'Tsid':[str(task_server_id)],
                    'Model':[],
                    'Torch Compile':[],
                    'Cache': [],
                    'NextN': [],
                    'Cuda Graph': [],
                    'Backend': [],
                    'Parallelism': []
                  }
    for key, value in server_args.items():
        if key == 'Model':
            if re.search(r"--model-path\s+(\S+)", command):
                model_str = ''
                if "DeepSeek-R1-BF16" in re.search(r"--model-path\s+(\S+)", command).group(1):
                    model_str += 'DS-R1-BF16'
                elif "DeepSeek-R1-W8A8" in re.search(r"--model-path\s+(\S+)", command).group(1):
                    model_str += 'DS-R1-W8A8'
            value.append(model_str)
        if key == 'Cache':
            value.append("HiCache" if re.search(r"--enable-hierarchical-cache+", command) else "RadixCache")
        elif key == 'Torch Compile':
            value.append("ON" if re.search(r"--enable-torch-compile+", command) else "OFF")
        elif key == 'NextN':
            value.append("ON" if re.search(r'--speculative-algo\s+NEXTN', command) else "OFF")
        elif key == 'Cuda Graph':
            value.append("OFF" if re.search(r"--disable-cuda-graph+", command) else "ON")
        elif key == 'Backend':
            if re.search(r"--attention-backend\s+(\S+)", command):
                value.append(re.search(r"--attention-backend\s+(\S+)", command).group(1))
            elif re.search(r"--enable-flashmla", command):
                value.append("flashmla")
            else:
                value.append('default')
        elif key == 'Parallelism':
            para_str = ""
            tp_size_match = re.search(r"--(tp|tp-size)\s+(\d+)",command)
            ep_size_match = re.search(r"--(ep|ep-size)\s+(\d+)",command)
            dp_size_match = re.search(r"--(dp|dp-size)\s+(\d+)",command)

            tp_str = ''
            dp_str = ''
            ep_str = ''

            assert tp_size_match
            tp_size = tp_size_match.group(2)
            tp_str = f"TP{tp_size}"

            if dp_size_match:
                dp_size = dp_size_match.group(2)
                dp_str = f"DP{dp_size}" 
                if re.search(r"--enable-dp-attention", command):
                    tp_str = f"TP{int(int(tp_size)/int(dp_size))}" 

            if ep_size_match:
                ep_str = f"EP{ep_size_match.group(2)}"
            elif re.search(r"--enable-ep-moe", command):
                ep_str = f"EP{tp_size}"

            value.append(f'{tp_str}{dp_str}{ep_str}')

    # output_string = ""
    # for key, value in server_args.items():
    #     output_string += f'__{key}-{value[0]}'
    # output_string = output_string.replace(" ", "")
    output_string = get_launch_server_args_str(command)
    return server_args, output_string

def get_launch_server_args_str(command):
    output_string = ""
    output_string += "-TCON" if re.search(r"--enable-torch-compile+", command) else "-TCOFF"
    output_string += "-NEXTNON" if re.search(r'--speculative-algo\s+NEXTN', command) else "-NEXTNOFF"
    output_string += "-CGOFF" if re.search(r"--disable-cuda-graph+", command) else "-CGON"
    output_string += "-HIERAR" if re.search(r"--enable-hierarchical-cache+", command) else "-RADIX"

    if re.search(r"--attention-backend\s+(\S+)", command):
        output_string += "-" + re.search(r"--attention-backend\s+(\S+)", command).group(1)
    if re.search(r"--enable-flashinfer-mla", command):
        output_string += "-flinfmla"

    if re.search(r"--enable-flashmla", command):
        output_string += "-flashmla"

    
    output_string += "-epmoe" if re.search(r"--enable-ep-moe", command) else ""
    output_string += "-dpatt" if re.search(r"--enable-dp-attention", command) else ""

    tp_size_match = re.search(r"--(tp|tp-size)\s+(\d+)",command)
    ep_size_match = re.search(r"--(ep|ep-size)\s+(\d+)",command)
    dp_size_match = re.search(r"--(dp|dp-size)\s+(\d+)",command)
    if tp_size_match:
        output_string += f"-tp{tp_size_match.group(2)}" 
    if ep_size_match:
        output_string += f"-ep{ep_size_match.group(2)}" 
    if dp_size_match:
        output_string += f"-dp{dp_size_match.group(2)}" 
    
    return output_string

def generate_machine(data,port):
    context = zmq.Context()
    slave_socket = []
    servers = data['servers']
    for server in data['servers']:
        print(f"IP: {server['ip']}, Port: {server['port']}, Username: {server['username']}, Password: {server['password']}")
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{server['ip']}:{port}")
        slave_socket.append(socket)

    # slave_ip_list = ['127.0.0.1']
    # slave_socket = []
    # for slave_ip in slave_ip_list:
    #     socket = context.socket(zmq.REQ)
    #     socket.connect(f"tcp://{slave_ip}:9999")
    #     slave_socket.append(socket)
    return slave_socket

def node_rank_command(cmd, rank):
    cmd += f" --node-rank {str(rank)} "
    return cmd

def generate_commands(data,args):
    task_list = []
    task_server_id = 0
    for task in data['tasks']:
        task_name = task['task_name']
        task_type = task['task_type']
        total_nodes_num = task['TOTAL_NODES_NUM']
        time_out = TIMEOUT_DURATION
        if "TIME_OUT" in task:
            time_out = task["TIME_OUT"]
        environment_variables = task.get('environment_variables', {}) 
        launch_server_commands = []
        benchmark_commands = []

        # launch_server
        launch_base = task['launch_server']['command_base']
        launch_variations = task['launch_server']['variations']

        launch_variation_params = list(launch_variations.values())
        launch_variation_names = list(launch_variations.keys())

        for combo in product(*launch_variation_params):
            full_command = launch_base
            for param in combo:
                full_command += f" {param.strip()}"
            full_command = full_command.strip()
            launch_server_commands.append(full_command)

        # benchmark
        benchmark_base = task['bench_serving']['command_base']
        benchmark_variations = task['bench_serving']['variations']

        benchmark_variation_params = list(benchmark_variations.values())
        benchmark_variation_names = list(benchmark_variations.keys())

        for combo in product(*benchmark_variation_params):
            full_command = benchmark_base
            for param in combo:
                full_command += f" {param.strip()}"
            full_command = full_command.strip()
            benchmark_commands.append(full_command)

        for launch_full_command in launch_server_commands:
            launch_server_args_dict,launch_server_args_str = get_launch_server_args(task_server_id,launch_full_command)
            task_dict = {
                'task_name': " ".join((task_name + launch_server_args_str).split()),
                'task_server_args':launch_server_args_dict,
                'task_type': task_type,
                'environment_variables': environment_variables,
                'TOTAL_NODES_NUM': int(total_nodes_num),
                "TIME_OUT":int(time_out),
                'launch_server': launch_full_command,
                'bench_serving': benchmark_commands
            }
            run_bench_serving(args.target_path, args.result_path, task_dict,init_result_file=True)
            task_list.append(task_dict)
            task_server_id += 1
    return task_list

def extract_metrics_from_file(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        command_match = re.search(r'^Command:\s*(.+)$', content, re.MULTILINE)
    bench_args = {
                    'batch-size':[],
                    'in-out':[],
                 }
    if command_match:
        command = command_match.group(1)
    else:
        command = ' '
    input_len,output_len,num_prompt = get_bench_serving_args(command)
    bench_args['in-out'].append(f'{input_len}-{output_len}')
    bench_args['batch-size'].append(f'{num_prompt}')

    metrics = {
        'Successful requests':[int(re.search(r'Successful requests:\s+(\d+)', content).group(1)) if re.search(r'Successful requests:\s+(\d+)', content) else 'None'],
        'Benchmark duration (s)': [float(re.search(r'Benchmark duration \(s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Benchmark duration \(s\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Total input tokens': [int(re.search(r'Total input tokens:\s+(\d+)', content).group(1)) if re.search(r'Total input tokens:\s+(\d+)', content) else 'None'],
        'Total generated tokens': [int(re.search(r'Total generated tokens:\s+(\d+)', content).group(1)) if re.search(r'Total generated tokens:\s+(\d+)', content) else 'None'],
        'Total generated tokens (retokenized)': [int(re.search(r'Total generated tokens \(retokenized\):\s+(\d+)', content).group(1)) if re.search(r'Total generated tokens \(retokenized\):\s+(\d+)', content) else 'None'],
        'Request throughput (req/s)': [float(re.search(r'Request throughput \(req/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Request throughput \(req/s\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Input token throughput (tok/s)': [float(re.search(r'Input token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Input token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Output token throughput (tok/s)': [float(re.search(r'Output token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Output token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Total token throughput (tok/s)': [float(re.search(r'Total token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Total token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Concurrency': [float(re.search(r'Concurrency:\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Concurrency:\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Mean E2E Latency (ms)': [float(re.search(r'Mean E2E Latency \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Mean E2E Latency \(ms\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Median E2E Latency (ms)': [float(re.search(r'Median E2E Latency \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Median E2E Latency \(ms\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Mean TTFT (ms)': [float(re.search(r'Mean TTFT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Mean TTFT \(ms\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Median TTFT (ms)': [float(re.search(r'Median TTFT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Median TTFT \(ms\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'P99 TTFT (ms)': [float(re.search(r'P99 TTFT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'P99 TTFT \(ms\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Mean TPOT (ms)': [float(re.search(r'Mean TPOT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Mean TPOT \(ms\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Median TPOT (ms)': [float(re.search(r'Median TPOT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Median TPOT \(ms\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'P99 TPOT (ms)': [float(re.search(r'P99 TPOT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'P99 TPOT \(ms\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Mean ITL (ms)': [float(re.search(r'Mean ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Mean ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'Median ITL (ms)': [float(re.search(r'Median ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Median ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else 'None'],
        'P99 ITL (ms)': [float(re.search(r'P99 ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'P99 ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else 'None'],
    }

    bench_result_data = {**bench_args, **metrics}

    return bench_result_data

def write_to_csv(metrics_list, output_file):
    headers = [
        'Successful requests', 'Benchmark duration (s)', 'Total input tokens',
        'Total generated tokens', 'Total generated tokens (retokenized)',
        'Request throughput (req/s)', 'Input token throughput (tok/s)',
        'Output token throughput (tok/s)', 'Total token throughput (tok/s)',
        'Concurrency', 'Mean E2E Latency (ms)', 'Median E2E Latency (ms)',
        'Mean TTFT (ms)', 'Median TTFT (ms)', 'P99 TTFT (ms)', 'Mean TPOT (ms)',
        'Median TPOT (ms)', 'P99 TPOT (ms)', 'Mean ITL (ms)', 'Median ITL (ms)',
        'P99 ITL (ms)'
    ]
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for metrics in metrics_list:
            writer.writerow(metrics)

def get_result(server_args,result_path):
    os.chdir(result_path)
    txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    result_df = pd.DataFrame()
    for txt_file in txt_files:
        metrics = extract_metrics_from_file(txt_file)
        server_args_df = pd.DataFrame(server_args)
        bench_result_df = pd.DataFrame(metrics)
        merge_df = pd.concat([server_args_df,bench_result_df],axis=1)
        result_df = pd.concat([result_df,merge_df],ignore_index=True)
    with open('benchmark_result.csv','w',encoding='utf-8') as csv_file:
        result_df.to_csv(csv_file, index=False)
        print(f"result_csv store in benchmark_result.csv")


def merge_result(target_path: str, result_path: str) -> None:
    try:
        result_path = os.path.join(target_path, result_path)
        all_data = pd.DataFrame()
        for subdir, dirs, files in os.walk(result_path):
            if subdir == result_path:
                continue
            dir_name = os.path.basename(subdir)
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(subdir,file)
                    data = pd.read_csv(file_path)
                    all_data = pd.concat([all_data, data], ignore_index=True)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_filename = f"{now}_result.csv"
        output_path = os.path.join(result_path, output_filename)
        with open(output_path,'w',encoding='utf-8') as csv_file:
            all_data.to_csv(csv_file, index=False)
        print(f"result_csv store in {output_path}")
    except Exception as e:
        print(f"merge_result exception {e}")
        pass


# ########################################################################### data processing ###########################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=str, default="benchmark/benchmark0421/result/", help="Path for storing results")
    parser.add_argument("--json-file", type=str, default="benchmark/benchmark0428/HQacc0428test.json", help="JSON file describing the task list")
    parser.add_argument("--progress-file", type=str, default="/pde_ai/share/sgl_automation/code/043.txt", help="real progress file")
    parser.add_argument("--target-path", type=str, default="/pde_ai/share/sgl_automation/", help="Target work path")
    parser.add_argument("--specify-task", type=bool, default=False, help="Starting from the designated task")
    parser.add_argument("--launchserver-id", type=int, default=0, help="Starting from launchserver-id")
    parser.add_argument("--benchserving-id", type=int, default=0, help="Starting benchserving-id")
    parser.add_argument("--current-port",type=int,default=9998,help="client port bind to recv msg")
    args = parser.parse_args(sys.argv[1:])
    return args


def run():
    global g_result_status, TOTAL_NODES_NUM,TIMEOUT_DURATION
    global CHECK_TIMEOUT_ON, CHECK_OTHER_ABNORMAL_ON
    finish_flag = ''
    args = get_args()
    target_directory = args.target_path
    result_path = args.result_path
    json_file_path = args.json_file

    create_real_progress_file(target_directory,args.result_path)
    create_log_file(target_directory,args.result_path)

    configure_logger(log_file=LOG_FILE)

    all_task_list, slave_machine_list = parse_json(json_file_path,args)
    if args.specify_task:
        g_result_status = ResultStatus(launch_server_id=args.launchserver_id, bench_serving_id=args.benchserving_id, test_result='')

    profile_task = False
    try:
        while not "all finishi" == finish_flag:
            task = all_task_list[g_result_status.launch_server_id]
            TOTAL_NODES_NUM = task['TOTAL_NODES_NUM']
            TIMEOUT_DURATION = task['TIME_OUT']
            
            
            if task["task_type"] == "profile":
                profile_path = set_profile_path(task['task_name'],task["environment_variables"])
                if not profile_task:
                    init_profile_bash()
                    profile_task = True

            with open(os.path.join(target_directory, REAL_PROGRESS_FILE), 'a') as f:
                print(f"\nRATE_OF_PROGRESS: {g_result_status.launch_server_id + 1}/{len(all_task_list)}", file=f)
                print(f"Task: {task['task_name']}", file=f)
                print(f"    Launch server: {task['launch_server']}", file=f)
                print(f"    Bench serving: {task['bench_serving']}\n", file=f)
            set_env(task["environment_variables"],slave_machine_list)

            while True:
                write_next_task(all_task_list,task,args)

                # run_master_launch_server(target_directory, node_rank_command(task['launch_server'], 0))
                # time.sleep(999999)
                # slave_cmd_map = run_child_launch_server(task['launch_server'], slave_machine_list)
                # stop_all(task, slave_cmd_map)
                # time.sleep(10)
                # exit_slave(slave_machine_list)
                # return
                
                # master test
                run_master_launch_server(target_directory, node_rank_command(task['launch_server'], 0))
                time.sleep(10)
                slave_cmd_map = run_child_launch_server(task['launch_server'], slave_machine_list)
                if not CHECK_OTHER_ABNORMAL_ON:
                    start_check_other_abnormal(task, slave_cmd_map)
                    pass

                run_bench_flag = False
                if wait_server_ready():
                    bench_statu, run_bench_flag = run_bench_serving(target_directory, result_path, task)
                    if run_bench_flag and profile_task:
                        wait_profile_finish(task["environment_variables"])
                        set_profile_bash(profile_path)
                stop_all(task, slave_cmd_map)
                time.sleep(10)
                finish_flag = update_result_status(all_task_list, task, run_bench_flag,args)

                logger.debug(f"----------------finish_flag : {finish_flag}")
                if not "this continue" == finish_flag:
                    break
                
        exit_slave(slave_machine_list)
        for sock in slave_machine_list:
            sock.close()

        merge_result(target_directory, result_path)
    finally:
        exit_master()
        for i in range(5):
            print("all_test_finish!!!")
        sys.exit(0)

def all_test_finish():
    # print("++++++++++++++subprocess.run++++++++++++++++")
    # chmod_result = subprocess.run(["chmod", "777", "-R", "/pde_ai/share/sgl_automation/"], timeout=5,capture_output=True, text=True)
    # print("++++++++++++++subprocess.end++++++++++++++++")
    # if chmod_result.returncode != 0:
    #     print(f"chmod error: {chmod_result.stderr}")
    for i in range(5):
        print("all_test_finish!!!")


if __name__ == "__main__":
    # get_result_test()
    run()
    all_test_finish()

