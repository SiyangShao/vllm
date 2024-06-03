from vllm import LLM

llm = LLM(model="facebook/opt-125m", enforce_eager=True,
          checkpoint_dir="./models/facebook/opt-125m", load_only=True)
