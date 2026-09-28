from transformers import AutoTokenizer, AutoModelForCausalLM
from bio_agent.profiling import memory
from bio_agent.dispatcher import execute
from bio_agent.utils import ask_model, summarize_with_llm
from peft import PeftModel
from bio_agent.config import ram_profiling, classifier_model_path, summarizer_model_path, base_model_path


# load_models

base = AutoModelForCausalLM.from_pretrained(base_model_path)
memory("Base model loaded", active=ram_profiling)
    
model = PeftModel.from_pretrained(base, classifier_model_path, adapter_name="classifier", device_map="auto", torch_dtype="auto")
memory("Classifier LoRA loaded", active=ram_profiling)

model.load_adapter(planner_model_path, adapter_name="planner")
memory("Planner LoRA loaded", active=ram_profiling)

model.load_adapter(summarizer_model_path, adapter_name="summarizer")
memory("Summarizer LoRA loaded", active=ram_profiling)    

tokenizer = AutoTokenizer.from_pretrained(classifier_model_path)
memory("Tokenizer loaded", active=ram_profiling)


def main():

    
    while True:
        try:
            original_prompt = input(">>> ").strip()

            if original_prompt.lower() in {"exit", "quit"}:
                break

            tool_prompt = f"User: {original_prompt}\nAssistant: "

            # PLAN
            model.set_adapter("planner")
            plan_graph = ask_model(
                tool_prompt,
                model,
                tokenizer,
            )
            print("\nPlan: ", plan_graph)
    

            # EXECUTE PLAN
            tool_results = execute(plan_graph)
            print("\nTool Results", tool_results)

            # VALIDATOR / SECURITY GUARD

            
            # SYNTHESIZER
            model.set_adapter("summarizer")
            response = summarize_with_llm(
                original_prompt,
                tool_results,
                model,
                tokenizer,
            )

            print("\nAssistant: ", response)
            
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()

