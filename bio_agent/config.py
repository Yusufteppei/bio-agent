import os
from pathlib import Path
from transformers import AutoTokenizer
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"


DATASET_ROOT = Path(".").resolve().parent / "datasets"
NOTEBOOK_ROOT = Path(".").resolve().parent / "notebooks"

BASE_MODEL_ID = base_model_path = "Qwen/Qwen2.5-0.5B"

classifier_model_path = "_checkpoints/qwen2.5-0.5b/v0.0.2"
summarizer_model_path = "_checkpoints/qwen2.5-0.5b/summarizer-0.0.1"


base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = tokenizer.eos_token

ram_profiling = True

