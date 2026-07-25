# Biology Tool Calling Dataset v0.0.1

## Overview

This dataset is the first experimental release of a supervised fine-tuning (SFT) dataset for a biology-focused tool-calling language model.

The objective is to teach a base language model to:

1. Decide whether an external tool is required.
2. Select the correct biological tool.
3. Produce correctly formatted JSON tool calls.
4. Generate direct responses when no tool is required.

This release is intended for validating the complete SFT pipeline rather than maximizing model performance.

---

## Version

**Dataset Version:** `0.0.1`

**Status:** Experimental

---

## Statistics

| Property          |  Value |
| ----------------- | -----: |
| Training Samples  |     30 |
| Test Samples      |      0 |
| Generation Method | GPT-4o |
| Format            |  JSONL |

---

## Available Tools

### `gene_lookup`

Retrieve information about a gene.

Example questions:

* Tell me about BRCA1.
* What is TP53?
* Give information on EGFR.

Arguments

```json
{
    "gene_name": "BRCA1"
}
```

---

### `search_pubmed`

Search the biomedical literature.

Example questions:

* Find papers about CRISPR.
* Search for Alzheimer's biomarkers.
* Recent work on protein folding.

Arguments

```json
{
    "query": "CRISPR gene editing"
}
```

---

### `classify_sequence`

Classify a biological sequence.

Example questions:

* What protein does this amino acid sequence correspond to?
* Classify this sequence.
* Predict the family of this protein.

Arguments

```json
{
    "sequence": "MKVLWAALLVTFLAGCQAKVE"
}
```

---

### `final_answer`

Used when no external tool is required.

Instead of producing a tool call, the assistant responds directly.

---

## Data Format

Each line of the dataset is a single JSON object.

Example:

```json
{
    "messages": [
        {
            "role": "user",
            "content": "Tell me about BRCA1."
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "tool": "gene_lookup",
                    "arguments": {
                        "gene_name": "BRCA1"
                    }
                }
            ]
        }
    ]
}
```

---

## Intended Use

This dataset is designed for:

* Supervised Fine-Tuning (SFT)
* LoRA / PEFT training
* Tool-calling experiments
* Biology agent development

It is **not** intended to teach biological knowledge. The base language model is expected to contain the underlying knowledge; this dataset teaches the model how and when to invoke tools.

---

## Known Limitations

* Very small dataset (30 samples).
* Generated automatically using GPT-4o.
* Limited tool coverage.
* Responses have not been exhaustively quality assured.
* Tool argument diversity is limited.

---

## Planned Improvements

* Increase dataset size to hundreds and eventually thousands of examples.
* Improve argument diversity.
* Reduce unnecessary null arguments.
* Add additional biology tools.
* Improve prompt diversity.
* Introduce multi-tool workflows.
* Add human quality assurance.

---

## Changelog

### v0.0.1

* Initial dataset release.
* 30 generated training examples.
* Four supported actions:

  * `gene_lookup`
  * `search_pubmed`
  * `classify_sequence`
  * `final_answer`
* Initial JSON schema established.
