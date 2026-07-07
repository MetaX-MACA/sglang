import torch


def weak_ref_tensor(tensor):
    return (
        torch.ops.sgl_kernel_custom.weak_ref_tensor(tensor)
        if isinstance(tensor, torch.Tensor)
        else tensor
    )
