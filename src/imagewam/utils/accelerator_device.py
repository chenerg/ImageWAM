import importlib
import os

import torch


def _load_torch_npu() -> None:
    if getattr(torch, "npu", None) is not None:
        return
    if importlib.util.find_spec("torch_npu") is None:
        return
    importlib.import_module("torch_npu")


def npu_is_available() -> bool:
    _load_torch_npu()
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and npu.is_available())


def manual_seed_all_accelerators(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if npu_is_available():
        torch.npu.manual_seed_all(seed)


def synchronize_device(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu" and npu_is_available():
        torch.npu.synchronize(device)


def set_accelerator_device(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu" and npu_is_available():
        torch.npu.set_device(device)


def resolve_train_device() -> str:
    if npu_is_available():
        device_count = torch.npu.device_count()
        if device_count <= 1:
            return "npu:0"
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank < 0 or local_rank >= device_count:
            return "npu:0"
        return f"npu:{local_rank}"

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return "cuda:0"
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank < 0 or local_rank >= device_count:
            return "cuda:0"
        return f"cuda:{local_rank}"

    return "cpu"
