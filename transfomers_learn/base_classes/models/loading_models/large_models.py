from transformers import AutoModel
import tempfile
import os

model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

import json

with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir, max_shard_size="1GB")
    with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "r") as f:
        index = json.load(f)

print("Model saved with index:", json.dumps(index, indent=2))