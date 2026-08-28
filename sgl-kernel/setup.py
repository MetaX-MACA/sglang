# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
import os
import setuptools
import platform
import re

def get_torch_version():
    """Get the current torch version."""
    try:
        import torch
        torch_version = torch.__version__
        version = re.sub(r'\+.*$', '', torch_version)
        version ="torch" + version
        return version
    except ImportError:
        return "unknown"

def get_platform_info():
    """Get platform information for wheel naming."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Normalize platform names
    if system == "linux":
        system = "linux"
    elif system == "darwin": 
        system = "macos"
    elif system == "windows":
        system = "win"
    
    # Normalize architecture names
    if machine in ["x86_64", "amd64"]:
        arch = "x86_64"
    elif machine in ["aarch64", "arm64"]:
        arch = "aarch64"
    elif machine.startswith("arm"):
        arch = "arm"
    else:
        arch = machine
    
    return f"{system}_{arch}"


def get_maca_version():
    """
    Returns the MACA SDK Version
    """
    maca_path = str(os.getenv('MACA_PATH'))
    if not os.path.exists(maca_path):
        return None
    file_full_path = os.path.join(maca_path, 'Version.txt')
    if not os.path.isfile(file_full_path):
        return None
    
    with open(file_full_path, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()
    return first_line.split(":")[-1]

# def get_maca_version_list():
#     version_str = get_maca_version()
#     version_list = list(map(int, (version_str or "0.0.0.0").split('.')))
#     version_list.extend([0] * (4 - len(version_list)))
#     return version_list


# This is to make sure that the package supports editable installs
version = "0.4.2.post2"
maca_ai_version = get_maca_version()
version += "+maca"+maca_ai_version
version += "."+get_torch_version()
setuptools.setup(version=version
)