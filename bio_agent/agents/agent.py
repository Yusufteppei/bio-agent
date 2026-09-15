from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from bio_agent.dispatcher import execute
from bio_agent.utils import ask_model, summarize_with_llm


class BioAgent:

    def __init__(self, model_path):

        self.model_path = Path(model_path)

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path)
        )

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path)
        )

    def tool_call(self, prompt):

        return ask_model(
            prompt,
            self.model,
            self.tokenizer,
        )

    def execute(self, tool_call_json):

        return execute(tool_call_json)

    def summarize(self, prompt, tool_results):

        return summarize_with_llm(
            prompt,
            tool_results,
            self.model,
            self.tokenizer,
        )

    def __call__(self, prompt):

        tool_call_json = self.tool_call(prompt)

        print("\nTool Call")
        print(tool_call_json)

        tool_results = self.execute(tool_call_json)

        print("\nTool Results")
        print(tool_results)

        return self.summarize(
            prompt,
            tool_results,
        )