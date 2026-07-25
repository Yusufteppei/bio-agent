import json
import inspect
from bio_agent.tools import *


TOOLS = {
    "search_pubmed": search_pubmed,
    "gene_lookup": gene_lookup,
    "classify_sequence": classify_sequence,
    "final_answer": final_answer
}




def execute(tool_call_json):

    tool_calls = json.loads(tool_call_json)
    results = []

    for call in tool_calls:

        tool_name = call["tool"]

        if tool_name not in TOOLS:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool = TOOLS[tool_name]

        signature = inspect.signature(tool)

        valid_args = {
            name: value
            for name, value in call.get("arguments", {}).items()
            if name in signature.parameters and value is not None
        }

        result = tool(**valid_args)

        results.append({
            "tool": tool_name,
            "result": result,
        })

    return results







