# Finetuning
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import pathlib

from transformers import set_seed
set_seed(42)

path = "./models/smollm2"
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=path)


# Load dataset
dataset = load_dataset("HuggingFaceTB/SmolLM2-135M-Instruct")

# Preprocess dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Split dataset into train and test sets
train_test_split = tokenized_dataset["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Ensure output directory exists
pathlib.Path("./finetuned/math-smollm2").mkdir(parents=True, exist_ok=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned/math-smollm2",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()
# Save the model
trainer.save_model("./finetuned/math-smollm2")
# Save the tokenizer
tokenizer.save_pretrained("./finetuned/math-smollm2")
# Save the training arguments

training_args.save("./finetuned/math-smollm2/training_args.bin")
