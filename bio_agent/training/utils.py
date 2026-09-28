import json
import os
import torch
import dotenv
import random
import re
from pathlib import Path
from openai import OpenAI
from bio_agent.config import base_tokenizer as tokenizer, BASE_MODEL_ID
dotenv.load_dotenv()

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


def generate_dataset(
    version: str,
    n: int,
    tools: list[dict],
    base_dir: str = "../../datasets",
    batch_size: int = 10,
):
    

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    dataset_dir = Path(base_dir) / version
    prompt_dir = dataset_dir / "prompts"

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompt_path = prompt_dir / "system_prompt.txt"
    tools_path = dataset_dir / "tools.json"
    output_file = dataset_dir / "train.jsonl"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Missing system prompt: {prompt_path}"
        )

    system_prompt = prompt_path.read_text(
        encoding="utf-8"
    )

    tools_path.write_text(
        json.dumps(
            tools,
            indent=4,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    tool_names = [
        tool["name"]
        for tool in tools
    ]

    print(f"Dataset: {version}")
    print(f"Output:  {output_file}")
    print(f"Tools:   {', '.join(tool_names)}")

    # ---------------------------------------------------------
    # Resume from existing valid JSONL lines.
    # ---------------------------------------------------------

    if output_file.exists():
        with open(
            output_file,
            "r",
            encoding="utf-8",
        ) as f:
            start = sum(
                1
                for line in f
                if line.strip()
            )
    else:
        start = 0

    print(f"Found {start} existing samples.")

    if start >= n:
        print("Dataset already complete.")
        return

    remaining = n - start

    # ---------------------------------------------------------
    # Generate in batches.
    # ---------------------------------------------------------

    with open(
        output_file,
        "a",
        encoding="utf-8",
    ) as f:

        while remaining > 0:

            current_batch = min(
                batch_size,
                remaining,
            )

            print(
                f"\nGenerating batch of "
                f"{current_batch} "
                f"({start + 1}-{start + current_batch})..."
            )

            prompt = f"""
                Generate exactly {current_batch} independent
                biology planning examples.
                
                Return ONLY JSONL.
                
                Each line must be one complete training example.
                Do not wrap the examples in an array.
                Do not use markdown fences.
                Do not add commentary.
                
                Vary:
                - biological topic
                - question complexity
                - number of DAG nodes
                - graph structure
                - tool usage
                - dependency structure
                
                Use the tools defined in the system prompt.
                """

            response = client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            text = response.output_text.strip()

            # -------------------------------------------------
            # Parse JSONL.
            # -------------------------------------------------

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            written = 0

            for line in lines:

                try:
                    obj = json.loads(line)

                    validate_example(obj)

                    f.write(
                        json.dumps(
                            obj,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    written += 1
                    start += 1

                    print(
                        f"{start}/{n} ✓"
                    )

                    # Checkpoint each accepted sample.
                    f.flush()

                except (
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as e:

                    print(
                        f"Invalid generated sample: {e}"
                    )

            remaining = n - start

            # -------------------------------------------------
            # Prevent infinite loops if model repeatedly fails.
            # -------------------------------------------------

            if written == 0:
                print(
                    "No valid samples generated. "
                    "Retrying batch..."
                )

    print(
        f"\nDataset complete: {start}/{n}"
    )


def validate_example(obj: dict):
    if not isinstance(obj, dict):
        raise ValueError(
            "Example must be an object."
        )

    messages = obj.get("messages")

    if not isinstance(messages, list):
        raise ValueError(
            "Missing messages."
        )

    if len(messages) != 2:
        raise ValueError(
            "Expected exactly 2 messages."
        )

    if messages[0].get("role") != "user":
        raise ValueError(
            "First message must be user."
        )

    if messages[1].get("role") != "assistant":
        raise ValueError(
            "Second message must be assistant."
        )

    content = messages[1].get("content")

    if not content:
        raise ValueError(
            "Assistant content is empty."
        )

    try:
        dag = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Assistant content is not valid JSON: {e}"
        )

    if "nodes" not in dag:
        raise ValueError(
            "DAG missing nodes."
        )

    if not isinstance(dag["nodes"], list):
        raise ValueError(
            "nodes must be a list."
        )

    ids = set()

    for node in dag["nodes"]:

        required = {
            "id",
            "tool",
            "args",
            "depends_on",
            "status",
            "progress",
        }

        missing = required - node.keys()

        if missing:
            raise ValueError(
                f"Node missing fields: {missing}"
            )

        if node["id"] in ids:
            raise ValueError(
                f"Duplicate node ID: {node['id']}"
            )

        ids.add(node["id"])

        if node["status"] is not None:
            raise ValueError(
                "Training status must be null."
            )

        if node["progress"] is not None:
            raise ValueError(
                "Training progress must be null."
            )

    # Check dependencies after collecting all IDs.
    for node in dag["nodes"]:

        for dependency in node["depends_on"]:

            if dependency not in ids:
                raise ValueError(
                    f"Unknown dependency: {dependency}"
                )

def validate_dataset(path: str | Path) -> list[tuple[int, str]]:
    """
    Validate every non-empty line in a JSONL dataset.

    Returns:
        List of (line_number, error_message) for failed samples.
    """
    path = Path(path)
    failures = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
                validate_example(obj)

            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as e:
                failures.append(
                    (line_number, str(e))
                )

    return failures