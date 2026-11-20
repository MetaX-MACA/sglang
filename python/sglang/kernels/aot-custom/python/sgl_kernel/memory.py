# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch


def weak_ref_tensor(tensor):
    return (
        torch.ops.sgl_kernel.weak_ref_tensor(tensor)
        if isinstance(tensor, torch.Tensor)
        else tensor
    )
