from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

# use the same API for 3 different tasks
# model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
# model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/SmolLM2-135M")
# model = AutoModelForQuestionAnswering.from_pretrained("HuggingFaceTB/SmolLM2-135M")

import os
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", token=os.environ.get("HF_TOKEN"), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=os.environ.get("HF_TOKEN"), trust_remote_code=True)