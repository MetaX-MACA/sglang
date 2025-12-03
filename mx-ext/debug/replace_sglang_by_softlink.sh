#!/bin/bash

workspace_dir="/opt/conda/lib/python3.10/site-packages"

if [ $# -ne 1 ]; then
    echo "usage: $0 <new sglang version dir>"
    exit 1
fi
code_dir=$1
if [ ! -d "${code_dir}" ]; then
    echo "${code_dir} not exist!"
    exit 1
fi

docker_sglang_version=`cat ${workspace_dir}/sglang/version.py`
new_replace_sglang_version=`cat ${code_dir}/version.py`
if [ "$docker_sglang_version" = "$new_replace_sglang_version" ]; then
    echo "## sglang version equal, can replace code!"
else
    echo "## docker sglang version $docker_sglang_version not equal to $new_replace_sglang_version, cannot replace!!"
    exit 1
fi


if [ -d "${workspace_dir}/sglang_origin" ]; then
    rm -rf ${workspace_dir}/sglang
else
    mv ${workspace_dir}/sglang ${workspace_dir}/sglang_origin
fi

ln -s ${code_dir} ${workspace_dir}/sglang
ls -l ${workspace_dir} | grep sglang

