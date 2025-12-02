import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,  # Efficient on modern GPUs
    device_map="auto",
    trust_remote_code=True,
)

# Define input text
prompt = "Explain quantum computing in simple terms."
messages = [{"role": "user", "content": prompt}]

# Tokenize input and generate response
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=500,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
