# Biology Tool-Calling Agent

A research project exploring supervised fine-tuning (SFT) for biological tool calling using open-source large language models.

The primary objective is to build a language model that can recognize when external biological tools are required, generate structured tool calls, execute those tools, and return accurate natural-language responses.

---

## Project Goals

* Learn the complete supervised fine-tuning (SFT) pipeline.
* Understand parameter-efficient fine-tuning (LoRA / PEFT).
* Build a production-style tool-calling agent from scratch.
* Explore biological AI applications through practical implementations.
* Develop a reusable framework for adding new biological tools.

---

## Current Features

* Supervised fine-tuning using LoRA.
* JSON-based tool calling.
* Automatic tool dispatching.
* Natural-language response generation from tool outputs.
* Dataset generation using GPT-4o.
* Versioned datasets and checkpoints.

---

## Supported Tools

Current tools include:

* `gene_lookup`
* `search_pubmed`
* `classify_sequence`
* `final_answer`

Additional tools will be added in future dataset releases.

---

## Project Structure

```text
bio-agent/
│
├── datasets/
│   └── v0.0.1/
│       ├── README.md
│       ├── metadata.json
│       ├── train.jsonl
│       └── test.jsonl
│
├── checkpoints/
│   └── qwen2.5-0.5b/
│       └── v0.0.1/
│
├── scripts/
│   ├── generate_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── tools/
│
└── README.md
```

---

## Workflow

```text
User
  │
  ▼
Fine-tuned LLM
  │
  ▼
JSON Tool Call
  │
  ▼
Python Dispatcher
  │
  ▼
Biological Tool
  │
  ▼
Structured Result
  │
  ▼
LLM Summarization
  │
  ▼
Final Answer
```

---

## Dataset Versioning

Datasets are versioned independently from model checkpoints.

Each dataset version contains:

* Training data
* Test data
* Metadata
* Documentation

Example:

```text
datasets/
    v0.0.1/
    v0.1.0/
    v1.0.0/
```

---

## Checkpoint Versioning

Each trained model is stored separately.

Example:

```text
checkpoints/
    qwen2.5-0.5b/
        v0.0.1/
        v0.1.0/

    qwen2.5-3b/
        v0.1.0/
```

Every checkpoint includes the adapter, merged model (if applicable), tokenizer, and metadata describing the training configuration.

---

## Roadmap

### Dataset

* Increase dataset size
* Improve prompt diversity
* Human quality assurance
* Add multi-tool examples
* Expand biological coverage

### Tools

Planned additions include:

* Protein lookup
* BLAST sequence search
* Protein structure prediction
* Variant lookup
* Pathway lookup
* GO annotation
* Disease association
* Sequence alignment

### Models

* Larger Qwen models
* Multi-tool agents
* Tool planning
* Multi-step reasoning
* Reinforcement learning experiments

---

## Motivation

This repository serves both as a learning project and as a foundation for future research in biology-focused AI agents. Rather than relying on general-purpose assistants, the long-term goal is to build specialized systems capable of interacting with domain-specific biological resources through reliable tool use.

---

## Current Status

| Component                   | Status |
| --------------------------- | ------ |
| Dataset Generation          | ✅      |
| SFT Pipeline                | ✅      |
| LoRA Training               | ✅      |
| Tool Calling                | ✅      |
| Tool Dispatch               | ✅      |
| Response Generation         | ✅      |
| Biological Tool Integration | 🚧     |
| Evaluation Suite            | 🚧     |
| Multi-tool Reasoning        | ⏳      |

---

## License

This project is intended for research and educational purposes.
