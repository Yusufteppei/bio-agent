from transformers import AutoTokenizer, AutoModelForCausalLM

from bio_agent.dispatcher import execute
from bio_agent.utils import ask_model, summarize_with_llm


MODEL_PATH = "_checkpoints/qwen2.5-0.5b/exp001/merged"


def load_model(model_path=MODEL_PATH):

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def main():

    model, tokenizer = load_model()

    while True:

        prompt = input(">>> ").strip()

        prompt = f"User: {prompt}\nAssistant: "
        if prompt.lower() in {"exit", "quit"}:
            break

        tool_call_json = ask_model(
            prompt,
            model,
            tokenizer,
        )

        print("\nTool Call")
        print(tool_call_json)

        tool_results = execute(tool_call_json)

        print("\nTool Results")
        print(tool_results)

        response = summarize_with_llm(
            prompt,
            tool_results,
            model,
            tokenizer,
        )

        print("\nAssistant")
        print(response)


if __name__ == "__main__":
    main()



"""from pathlib import Path

from bio_agent.agent import BioAgent

MODEL_PATH = (
    Path(__file__).parent
    / "checkpoints"
    / "qwen2.5-0.5b"
    / "v0.0.1"
    / "merged"
)

agent = BioAgent(MODEL_PATH)

while True:

    prompt = input(">>> ")

    if prompt.lower() in {"quit", "exit"}:
        break

    print()
    print(agent(prompt))
    print()
"""