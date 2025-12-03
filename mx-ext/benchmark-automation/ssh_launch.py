import paramiko
import time
import sys
from datetime import datetime
import os
import json
import argparse
import subprocess
import psutil

LOG_FILE = None
g_finish_flag = False
g_last_update_time = time.time()

def create_log_file(target_directory):
    global LOG_FILE
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    LOG_FILE = os.path.join(target_directory, f"{'LOG_FILE_/LOG_FILE_'}_{now}.txt")
    create_file(LOG_FILE)

def create_file(filename):
    directory = os.path.dirname(filename)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(filename):
        with open(filename, "w") as file:
            pass

def kill_process_by_pid(pid):
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=3)  # 等待进程终止
        print(f"已杀死 PID 为 {pid} 的进程")
    except psutil.NoSuchProcess:
        print(f"PID 为 {pid} 的进程不存在")
    except psutil.TimeoutExpired:
        print(f"PID 为 {pid} 的进程无法在规定时间内终止")
    except Exception as e:
        print(f"发生错误：{e}")


def run_sys_cmd(cmd: str):
    """Run |cmd| and return its output."""
    handle = subprocess.Popen([cmd], shell=True, bufsize=1, text=True, encoding='utf-8',
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        line = handle.stdout.readline()
        if not line and handle.poll() is not None:
            print(f"[{cmd}] exit")
            break
        print(line.strip())


class SSHController:
    def __init__(self, host, port=22, username=None, password=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.ssh_client = None
        self.pid = None
        self.stdin = None
        self.stdout = None
        self.stderr = None


    def connect(self):
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        private_key = paramiko.RSAKey.from_private_key_file('/remote_home/m01084/.ssh/id_rsa')
        try:
            self.ssh_client.connect(
                            hostname=self.host,    
                            username=self.username, 
                            pkey=private_key,       
                            look_for_keys=False,
                            allow_agent=False
                        )
            print("SSH connection established.")
            return True
        except Exception as e:
            print(f"Failed to connect to {self.host}: {e}")
            return False
        # try:
        #     self.ssh_client.connect(self.host, self.port, self.username, self.password)
        #     print("SSH connection established.")
        #     return True
        # except Exception as e:
        #     print(f"Failed to connect to {self.host}: {e}")
        #     return False

    def execute_command(self, command):
        command_to_execute = f"echo $$; exec {command}"
        # command_to_execute = f"{command}"
        stdin, stdout, stderr = self.ssh_client.exec_command(command_to_execute, get_pty=True)
        self.pid = int(stdout.readline().strip())
        print(f"Process {command_to_execute} started with PID: {self.pid}")

        # import threading
        # monitor_thread = threading.Thread(target=self.monitor_process, args=())
        # monitor_thread.daemon = True
        # monitor_thread.start()
        
        return self.pid,stdin, stdout, stderr

    def _read_output(self, stdout, ):
        global LOG_FILE,g_finish_flag,g_last_update_time
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                try:
                    # print(f"**************read_output****************")
                    output = stdout.channel.recv(2048).decode('utf-8', errors='ignore')
                    print(output, end='', flush=True)
                    g_last_update_time = time.time()
                    if "all_test_finish!!!" in output:
                        g_finish_flag = True
                except Exception as e:
                    print(f"_read_output exception {e}")
                    g_finish_flag = True
                # with open(LOG_FILE, 'a') as f:
                #     print(output, file=f)
        print("Subprocess has exited")
    
    def execute_command_with_realtime_output(self, command):
        pid , stdin, stdout, stderr = self.execute_command(command)

        import threading
        output_thread = threading.Thread(target=self._read_output, args=(stdout,))
        output_thread.daemon = True
        output_thread.start()
    
        return self.pid
        
        
    def monitor_process(self):
        print(f"Monitoring process with PID: {self.pid}")
        while True:

            _, stdout, _ = self.ssh_client.exec_command(f"ps -p {self.pid}")
            output = stdout.read().decode().strip()
            if self.pid in output:  
                print(f"Process {self.pid} is running.")
            else:  
                print(f"Process {self.pid} has exited.")
                break

            time.sleep(60)


    def kill_process(self):
        print(f"Killing process with PID: {self.pid}")
        _, stdout, stderr = self.ssh_client.exec_command(f"kill -9 {self.pid}")
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print(f"Process {self.pid} has been killed.")
        else:
            print(f"Failed to kill process {self.pid}: {stderr.read().decode()}")
    

    def close(self):
        if self.ssh_client:
            self.ssh_client.close()
            print("SSH connection closed.")


def connect_machine(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    machine_list = []
    servers = data['servers']
    for server in servers:
        print(f"IP: {server['ip']}, Port: {server['port']}, Username: {server['username']}, Password: {server['password']}")
        ssh_controller = SSHController(server['ip'], server['port'], server['username'], server['password'])
        if not ssh_controller.connect():
            print("SSH connection failed.")
            sys.exit(1)
        machine_list.append(ssh_controller)
    return machine_list


def machine_run_cmd(machine_list, commands):
    for machine in machine_list:
        for cmd in commands:
            machine.execute_command_with_realtime_output(cmd)
            time.sleep(1)

def launch_docker(machine_list, launch_docker_cmd):
    machine_run_cmd(machine_list, launch_docker_cmd)

def clear_machine(machine_list, clear_machine_cmd):
    machine_list = list(reversed(machine_list))
    machine_run_cmd(machine_list, clear_machine_cmd)

def clear_triton_cache(machine_list, launch_docker_cmd):
    machine_run_cmd(machine_list, launch_docker_cmd)

def run_benchmark_test(machine_list):
    COMMAND_TO_START = "docker exec -it sshtestdyl  /opt/conda/bin/python3 /pde_ai/share/sgl_automation/code/ttlaunch.py"  
    for machine in machine_list:
        machine.execute_command_with_realtime_output(COMMAND_TO_START)

def run_benchmark(machine_list,benchmark_cmd):
    pid_list = []
    master = True
    for machine, cmd in zip(machine_list,benchmark_cmd):
        if master:
            pid = machine.execute_command_with_realtime_output(cmd)
            pid_list.append(pid)
            master = False
        else:
            time.sleep(2)
            out = machine.execute_command(cmd)
            pid_list.append(out[0])
    return pid_list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container_name", type=str, default="043acc0507dyl", help="container name")
    parser.add_argument("--image_id", type=str, default="3fe763574f33", help="docker images id")
    parser.add_argument("--result-path", type=str, default="accuracy/accuracy0507/test_result/", help="Path for storing results")
    parser.add_argument("--json-file", type=str, default="accuracy/accuracy0507/HQacc0507.json", help="JSON file describing the task list")
    parser.add_argument("--progress-file", type=str, default="/pde_ai/share/sgl_automation/code/043.txt", help="real progress file")
    parser.add_argument("--target-path", type=str, default="/pde_ai/share/sgl_automation/", help="Target work path")
    parser.add_argument("--specify-task", type=bool, default=False, help="Starting from the designated task")
    parser.add_argument("--launchserver-id", type=int, default=0, help="Starting from launchserver-id")
    parser.add_argument("--benchserving-id", type=int, default=0, help="Starting benchserving-id")
    parser.add_argument("--current-port",type=int,default=9998,help="client port bind to recv msg")
    args = parser.parse_args(sys.argv[1:])
    return args

def run(args,machine_list):
    

    launch_docker_cmd = [f"docker run -it --net=host --uts=host --ipc=host --device=/dev/dri --device=/dev/mxcd  --device=/dev/infiniband --privileged=true \
                         --group-add video --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size 100gb --ulimit memlock=-1 \
                           --name {args.container_name}  \
                           -v /home/sw:/home/sw  \
                           -v /mnt/:/mnt/  \
                           -v /pde_ai/:/pde_ai/  \
                           {args.image_id}  \
                           /bin/bash", f"docker start {args.container_name}", f"docker exec -it {args.container_name} /bin/bash "]
    
    launch_docker(machine_list, launch_docker_cmd)

    clear_sys(args,machine_list)

    benchmark_cmd = [f"docker exec -it {args.container_name}  /bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh; conda activate base; cd /pde_ai/share/sgl_automation; \
                     python3 code/master_launch_profile_result_dev.py --json-file {args.json_file} --progress-file {args.progress_file} --result-path {args.result_path} --target-path {args.target_path}  \
                        --specify-task {args.specify_task} --launchserver-id {args.launchserver_id} --benchserving-id {args.benchserving_id} --current-port {args.current_port}'",
                    ]
    for rank in range(len(machine_list)):
        if rank == 0:
            continue
        cmd = f"docker exec -it {args.container_name}  /bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh; conda activate base; \
                cd /pde_ai/share/sgl_automation; python3 code/slave_launch.py  --current-ip {machine_list[rank].host} --current-port {args.current_port} --result-path {args.result_path} --target-path {args.target_path} '"
        benchmark_cmd.append(cmd)
    pid_list = run_benchmark(machine_list,benchmark_cmd)

    return args,machine_list

def clear_sys(args,machine_list):
    exit_cmd = [f"docker exec -it {args.container_name}  /bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh; conda activate base; cd /pde_ai/share/sgl_automation; python3 code/exit_sglang.py '"]
    machine_run_cmd(machine_list,exit_cmd)


if __name__ == "__main__":
    args = get_args()
    machine_list = connect_machine(os.path.join(args.target_path,args.json_file))
    try:
        run(args,machine_list)
        time_sleep = 60
        while not g_finish_flag:
            time.sleep(time_sleep)
            print('**************************ssh launch is running******************************')
            if time.time() - g_last_update_time > 1500:
                print('**************************ssh launch timeout******************************')
                clear_sys(args,machine_list)
                sys.exit(1)
    finally:
        clear_sys(args,machine_list)

    # run_sys_cmd("bash /pde_ai/share/sgl_automation/profile/run_test_bash.sh")