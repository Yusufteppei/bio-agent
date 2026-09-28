from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from transformers import AutoTokenizer, AutoModelForCausalLM
from bio_agent.profiling import memory
from bio_agent.dispatcher import execute
from bio_agent.utils import ask_model, summarize_with_llm
from peft import PeftModel
from bio_agent.config import ram_profiling, classifier_model_path, summarizer_model_path, base_model_path


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


class State(TypedDict):
    message: str
    plan: TypedDict


def planner(state: State) -> State:
    print("Planning...")
    model.set_adapter("planner")
    return state


def executor(state: State) -> State:
    print("Executing...")
    return state


def summarizer(state: State) -> State:
    print("Summarizing...")
    model.set_adapter("summarizer")
    return state


graph = StateGraph(State)

graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("validator", validator)


graph.add_edge(START, "validator")
graph.add_edge("validator", "planner")
graph.add_edge("planner", "validator")
graph.add_edge("validator", "executor")
graph.add_edge("validator", "planner")
graph.add_edge("executor", "validator")
graph.add_edge("validator", "summarizer")
graph.add_edge("summarizer", END)


app = graph.compile()

app.invoke({"message": "Find information about TP53"})