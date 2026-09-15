import os
import psutil
import torch

process = psutil.Process(os.getpid())

def memory(label, active=True):
    if not active:
        return

    ram = process.memory_info().rss / 1024**3

    print(f"\n{label}")
    print(f"RAM:  {ram:.2f} GB")

    if torch.cuda.is_available():
        print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"VRAM reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"VRAM peak:     {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")