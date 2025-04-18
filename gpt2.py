from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pathlib
import torch
import time
import sys

pathlib.Path("./models/gpt2").mkdir(parents=True, exist_ok=True)

path = "./models/gpt2"
model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=path)
# Set pad token to eos token (this is the recommended approach for GPT-2)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=path)


def generate_text(prompt, max_length=100):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate text
    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,  # Enable sampling
        temperature=0.7
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def pretty(prompt):
    result = generate_text(prompt)
    for i in range(len(result)):
        print(result[i], end='', flush=True)
        time.sleep(0.02)
    print()
    print("\n")
    print(f"Word Count: {len(result)}")
    print(result)

if __name__ == "__main__":
    pretty("Oh sweet Africa")