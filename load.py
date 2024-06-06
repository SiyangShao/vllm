from transformers import AutoTokenizer
import time
import torch
from checkpoint_store import load_model

start = time.time()
model = load_model("facebook/opt-125m", storage_path="./models/")
# Please note the loading time depends on the model size and the hardware bandwidth.
print(f"Model loading time: {time.time() - start:.2f}s")


