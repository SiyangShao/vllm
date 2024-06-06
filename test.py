from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m", load_format="checkpoint")
