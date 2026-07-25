import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datasets import load_dataset
import json

#from huggingface_hub import hf_hub_download


MODEL_ID = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)


model_1 = get_peft_model(model, lora_config)


dataset = load_dataset(
    "json",
    data_files="../datasets/v0.0.1/train.jsonl"
)


tokenized_dataset = dataset.map(
    tokenize,
    remove_columns=dataset['train'].column_names,
)

loader = DataLoader(
    dataset=tokenized_dataset['train'],
    batch_size=3,
    collate_fn=collate_fn
)


optimizer = torch.optim.Adam(params=model.parameters(),lr=3e-4)
loss = torch.nn.CrossEntropyLoss(ignore_index=-100)



NUM_EPOCHS = 4

model_1.train()

for epoch in range(NUM_EPOCHS):

    total_loss = 0.0

    for batch in loader:
        print("===")
        input_ids = batch["input_ids"].to(model_1.device)
        attention_mask = batch["attention_mask"].to(model_1.device)
        labels = batch["labels"].to(model_1.device)

        outputs = model_1(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}: "
        f"{total_loss / len(loader):.4f}"
    )

# ### Fine tuned model



model_1.save_pretrained("../_checkpoints/qwen2.5-0.5b/exp001")
tokenizer.save_pretrained("../_checkpoints/qwen2.5-0.5b/exp001")




# 1. Load the pristine, untouched base model
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# 2. Inject your saved fine-tuned adapters right back into it
model_1 = PeftModel.from_pretrained(base_model, "../_checkpoints/qwen2.5-0.5b/exp001")

