from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto")

input_text = "Il mio nome è"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

outputs = model.generate(input_ids, do_sample=True, max_length=50, num_return_sequences=5)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
