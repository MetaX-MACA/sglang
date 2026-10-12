"""Platform checks shared by the standalone kernel build and runtime."""



def is_maca(torch_version: str | None = None) -> bool:
    if torch_version is None:
        import torch

        torch_version = torch.__version__
    return "metax" in torch_version
