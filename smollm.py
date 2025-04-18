from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pathlib

pathlib.Path("./models/smollm2").mkdir(parents=True, exist_ok=True)

path = "./models/smollm2"
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=path)

# Inference function
def generate_text(prompt, max_length=100, device='cuda'):
    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set device
    device = torch.device(device)
    model.to(device)
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

print(pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)(
    "Hi, Name is Dan",
    max_length=50,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
)[0]["generated_text"])