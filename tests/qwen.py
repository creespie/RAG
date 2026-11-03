import time

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"

start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
print(f"Modello caricato in {time.time() - start:.1f} s")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python? Answer in one sentence."},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tokenizer(prompt, return_tensors="pt")

start = time.time()
output = model.generate(
    **inputs, max_new_tokens=100, do_sample=True, temperature=0.7,
    top_p=0.8, top_k=20
)
new_tokens = output[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
print(f"Risposta generata in {time.time() - start:.1f} s")
