from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkDetector, WatermarkingConfig

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

inputs = tokenizer(["This is the beginning of a long story", "Alice and Bob are"], padding=True, return_tensors="pt")
input_len = inputs["input_ids"].shape[-1]

watermarking_config = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
out = model.generate(**inputs, watermarking_config=watermarking_config, do_sample=False, max_length=50)

detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config=watermarking_config)
detection_out = detector(out, return_dict=True)
print(detection_out.prediction)