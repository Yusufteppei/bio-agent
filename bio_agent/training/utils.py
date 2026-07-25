if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_example(example):

    user = example["messages"][0]["content"]

    assistant = example["messages"][1]

    if "tool_calls" in assistant:

        assistant_text = json.dumps(
            assistant["tool_calls"],
            ensure_ascii=False
        )

    else:

        assistant_text = assistant["content"]

    return (
        f"User: {user}\nAssistant: ",
        f"{assistant_text}"
    )
#format_example(d1)

MAX_LENGTH = 512

def tokenize(example):

    prompt, answer = format_example(example)

    # Don't add BOS/EOS separately to each piece
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        answer + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]

    # Concatenate into one sequence
    input_ids = prompt_ids + answer_ids

    # Truncate if necessary
    input_ids = input_ids[:MAX_LENGTH]

    # Labels:
    # Ignore prompt, learn only assistant response
    labels = (
        [-100] * len(prompt_ids)
        + answer_ids
    )[:MAX_LENGTH]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

PAD_ID = tokenizer.pad_token_id

def collate_fn(batch):

    max_len = max(len(sample["input_ids"]) for sample in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for sample in batch:

        pad_len = max_len - len(sample["input_ids"])

        input_ids.append(
            sample["input_ids"] + [PAD_ID] * pad_len
        )

        attention_masks.append(
            sample["attention_mask"] + [0] * pad_len
        )

        labels.append(
            sample["labels"] + [-100] * pad_len
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }