import torch
from marker.settings import settings
import requests
import traceback
from typing import Any
import threading
from tracing import get_propagation_headers


def flush_cuda_memory():
    if settings.TORCH_DEVICE_MODEL == "cuda":
        torch.cuda.empty_cache()


def send_callback(callback_url: str, result: Any):
    headers = get_propagation_headers()
    threading.Thread(target=send_callback_inner, args=(callback_url, result, headers)).start()


def send_callback_inner(url: str, result: Any, headers: dict[str, str]):
    try:
        print('callback url: ', url, flush=True)
        response = requests.post(url, json=result, headers=headers, timeout=30)
        print(f"Callback response status: {response.text}", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"Callback failed: {e}", flush=True)
