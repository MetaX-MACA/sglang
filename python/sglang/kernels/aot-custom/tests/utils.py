# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch


def is_sm10x():
    return torch.cuda.get_device_capability() >= (10, 0)


def is_hopper():
    return torch.cuda.get_device_capability() == (9, 0)
