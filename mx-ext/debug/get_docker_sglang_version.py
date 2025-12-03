import os
import sys
import subprocess
import dataclasses
import argparse
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


def run_cmd(cmd: str) -> ProcStatus:
    proc = ProcStatus(store_output=True)
    run_sys_cmd(cmd, proc)


def get_docker_sglang_version(docker_name: str, image_id: str):
    stop_cmd = f'docker stop {docker_name}'
    run_cmd(stop_cmd)
    rm_cmd = f'docker rm {docker_name}'
    run_cmd(rm_cmd)
    
    print(f'## start run {docker_name=} {image_id=} on local !!')
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
    run_cmd(docker_start_cmd)

    full_cmd = f'docker exec {docker_name} /bin/bash -c "cat /opt/conda/lib/python3.10/site-packages/sglang/version.py"'
    run_cmd(full_cmd)

    stop_cmd = f'docker stop {docker_name}'
    run_cmd(stop_cmd)
    rm_cmd = f'docker rm {docker_name}'
    run_cmd(rm_cmd)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--image",
    type=str,
    required=True,
    help="image id need"
)
raw_args = parser.parse_args(sys.argv[1:])

docker_name = f'get_sglang_version_tmp'
get_docker_sglang_version(docker_name, raw_args.image)

