from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
# assistant_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, 
                        #  tokenizer=tokenizer,
                        #  assistant_tokenizer=assistant_tokenizer, 
                         assistant_model=assistant_model, 
                         do_sample=True, 
                         temperature=0.5)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(result)
# 'Hugging Face is an open-source company that is dedicated to creating a better world through technology.'