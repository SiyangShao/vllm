from checkpoint_store import save_model

# Load a model from HuggingFace model hub.
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')

# Replace './models' with your local path.
save_model(model, './models/facebook/opt-125m')
