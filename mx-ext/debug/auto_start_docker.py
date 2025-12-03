import os
import subprocess
import dataclasses
import socket
from typing import Optional, List, Dict

os.environ[ 'PYTHONUNBUFFERED' ]='1'

@dataclasses.dataclass
class ProcStatus:
    handle: Optional[None] = None
    output: Optional[List[str]] = None
    store_output: Optional[bool] = False
    print_output: Optional[bool] = True
    ready_flag: Optional[List[str]] = None
    is_ready: Optional[bool] = False


@dataclasses.dataclass
class SSHInfo:
    ip: Optional[str] = ''
    user: Optional[str] = ''
    passwd: Optional[str] = ''


@dataclasses.dataclass
class DockerPsInfo:
    container_id: Optional[str] = ''
    image: Optional[str] = ''
    command: Optional[str] = ''
    created: Optional[str] = ''
    status: Optional[str] = ''
    port: Optional[str] = ''
    name: Optional[str] = ''


def run_sys_cmd(cmd: str, proc: ProcStatus):
    """Run |cmd| and return its output."""
    print(f'## Run {cmd}')
    proc.handle = subprocess.Popen([cmd], shell=True, bufsize=1, text=True, encoding='utf-8',
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        line = proc.handle.stdout.readline()
        if not line and proc.handle.poll() is not None:
            # print(f"[{cmd}] exit")
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


def run_remote_sys_cmd(cmd: str, proc: ProcStatus, remote_ssh_info: SSHInfo):
    run_sys_cmd(f"ssh -o StrictHostKeyChecking=no {remote_ssh_info.user}@{remote_ssh_info.ip} '{cmd}'", proc)


def run_cmd(cmd: str, is_local: bool, remote_ssh_info: SSHInfo) -> ProcStatus:
    proc = ProcStatus(store_output=True)
    if is_local:
        run_sys_cmd(cmd, proc)
    else:
        run_remote_sys_cmd(cmd, proc, remote_ssh_info)
    return proc


def get_local_ip() -> str:
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


def get_docker_ps(is_local: bool, remote_ssh_info: SSHInfo) -> Dict[str, DockerPsInfo]:
    ps_cmd = 'docker ps -a'
    proc_status = run_cmd(ps_cmd, is_local, remote_ssh_info)
    docker_ps_name = {}
    split_pos = [0]
    begin = False
    for line in proc_status.output:
        if 'CONTAINER ID' in line:
            split_pos.append(line.find('IMAGE'))
            split_pos.append(line.find('COMMAND'))
            split_pos.append(line.find('CREATED'))
            split_pos.append(line.find('STATUS'))
            split_pos.append(line.find('PORTS'))
            split_pos.append(line.find('NAMES'))
            begin = True
            continue
        if not begin:
            continue
        if len(line) < split_pos[-1]:
            break
        attr_list = []
        for i, _ in enumerate(split_pos):
            if i + 1 == len(split_pos):
                attr_list.append(line[split_pos[i]:].strip())
            else:
                attr_list.append(line[split_pos[i]:split_pos[i + 1]].strip())

        if len(attr_list) != len(dataclasses.fields(DockerPsInfo)):
            print(f'## invalid docker ps line: {line}')
            break
        docker_info = DockerPsInfo(*attr_list)
        docker_ps_name[docker_info.name] = docker_info
    return docker_ps_name


def check_docker_exist(is_local: bool, docker_name: str, image_id: str, remote_ssh_info: SSHInfo, force_rm: bool=False) -> bool:
    dockers = get_docker_ps(is_local, remote_ssh_info)
    if docker_name not in dockers.keys():
        return False
    same_docker = dockers[docker_name]
    is_exit = 'Exited' in same_docker.status
    if same_docker.image == image_id and not force_rm:
        print(f'## check {docker_name} {image_id} exist, no need create new!!')
        if is_exit:
            start_cmd = f'docker start {docker_name}'
            run_cmd(start_cmd, is_local, remote_ssh_info)
        return True
    else:
        if force_rm:
            print(f'## check {docker_name} exist, but force rm , stop and rm {docker_name}!!')
        else:
            print(f'## check {docker_name} exist, but image id not equal(new:{image_id}, old:{same_docker.image} , stop and rm {docker_name}!!')
        if not is_exit:
            stop_cmd = f'docker stop {docker_name}'
            run_cmd(stop_cmd, is_local, remote_ssh_info)
        rm_cmd = f'docker rm {docker_name}'
        run_cmd(rm_cmd, is_local, remote_ssh_info)
        return False


def check_docker_alive(is_local: bool, docker_name: str, image_id: str, remote_ssh_info: SSHInfo) -> bool:
    dockers = get_docker_ps(is_local, remote_ssh_info)
    if docker_name not in dockers.keys():
        return False
    same_docker = dockers[docker_name]
    is_exit = 'Exited' in same_docker.status
    if same_docker.image != image_id:
        return False
    return not is_exit


def start_docker(is_local: bool, docker_name: str, image_id: str, remote_ssh_info: SSHInfo, prepare_docker_cmds: List, force_rm: bool=False):
    if check_docker_exist(is_local, docker_name, image_id, remote_ssh_info, force_rm):
        return
    
    print(f'## start run {docker_name=} {image_id=} on {"local" if is_local else remote_ssh_info.ip} !!')
    docker_start_cmd = f'docker run -it --net=host --uts=host --ipc=host --device=/dev/dri --device=/dev/mxcd --privileged=true --group-add video \
--security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size 100gb --ulimit memlock=-1 \
    -d \
    --name {docker_name} \
    -v /pde_ai/datasets:/pde_ai/datasets \
 -v /pde_ai/share:/pde_ai/share \
 -v /pde_ai/models:/pde_ai/models \
   -v /mnt/data:/mnt/data \
    {image_id} \
    /bin/bash'
    run_cmd(docker_start_cmd, is_local, remote_ssh_info)

    if not check_docker_alive(is_local, docker_name, image_id, remote_ssh_info):
        print(f'## start docker {docker_name=} {image_id=} on {"local" if is_local else remote_ssh_info.ip} faild !!')
        return
    

    for cmd in prepare_docker_cmds:
        if not cmd:
            continue
        full_cmd = f'docker exec {docker_name} /bin/bash -c ". /opt/conda/etc/profile.d/conda.sh;conda activate base;{cmd}"'
        run_cmd(full_cmd, is_local, remote_ssh_info)



# 放私有代码的地方，每个机器都可以访问 
private_share_dir = '/pde_ai/share/xinyang'

# ssh用户名，默认添加到docker名字上
ssh_user = 'xinyang'

# docker名称和镜像id
docker_name = f'v045_{ssh_user}'
docker_image = '0246c13f7240'

# 是否强制删除已有的同名镜像
force_rm_exist_docker = True
# 是否使用git下载的最新代码替换所有镜像的
use_git_code = False

if use_git_code:
    # 如果使用git下载的代码目录替换，直接设置好sglang目录，然后执行replace_code_cmd就行
    # code_dir 为已经down好代码的目录
    code_dir = f'{private_share_dir}/develop/sglang_v045/python/sglang'
    copy_docker_code_to_share_cmd = None
else:
    # 如果使用docker镜像自带的代码，执行这个
    code_dir = f'{private_share_dir}/{docker_name}_sglang'
    # 这个命令只需要在以下第一个节点执行即可
    copy_docker_code_to_share_cmd = f'cd {private_share_dir}; rm -rf {docker_name}_sglang; \
        cp -rf /opt/conda/lib/python3.10/site-packages/sglang {code_dir}'
replace_code_cmd = f'{private_share_dir}/tools/replace_sglang_by_softlink.sh {code_dir}'
copy_vscode_cmd = f'cp -rf {private_share_dir}/.vscode-server /root/'
start_slave_cmd = f'python {private_share_dir}/tools/slave.py &'

# prepare_deep_ep = f'pip install {private_share_dir}/deep_ep/deep_ep-1.0.0+b70c5f2-cp310-cp310-linux_x86_64.whl; \
#     cp -rf {private_share_dir}/deep_ep/mxdeepep_py310/mxshmem /opt/maca/'

docker_ips = {
    '10.2.179.91': [
        copy_docker_code_to_share_cmd, # 只需要在第一个节点执行
        replace_code_cmd,
        # copy_vscode_cmd,
    ],
    '10.2.179.92': [
        replace_code_cmd,
        # copy_vscode_cmd,
        # start_slave_cmd,
    ],
    '10.2.179.93': [
        replace_code_cmd,
        # copy_vscode_cmd,
        # start_slave_cmd,
    ],
    '10.2.179.94': [
        replace_code_cmd,
        # copy_vscode_cmd,
        # start_slave_cmd,
    ]
}

local_ip = get_local_ip()
if local_ip == '0.0.0.0':
    print("## unable to get local ip !")
    exit(1)

for ip, docker_cmds in docker_ips.items():
    print(f'#################################################################### start docker on {ip} ####################################################################')
    ssh_info = SSHInfo(ip=ip, user=ssh_user)
    is_local = ip == local_ip
    start_docker(is_local, docker_name, docker_image, ssh_info, docker_cmds, force_rm_exist_docker)

