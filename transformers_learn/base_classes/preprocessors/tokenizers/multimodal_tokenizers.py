from transformers import AutoTokenizer

vision_tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    extra_special_tokens={"image_token": "<image>", "boi_token": "<image_start>", "eoi_token": "<image_end>"}
)
print(vision_tokenizer.image_token, vision_tokenizer.image_token_id)
# ("<image>", 32000)

# vision_tokenizer.save_pretrained("./datasets/pipeline-cat-chonk.jpeg")