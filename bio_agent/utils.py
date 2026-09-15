import json
import torch

MAX_LENGTH = 512

def ask_model(prompt, model, tokenizer):
    model.eval()
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )
    
    
    generated = model.generate(
        **inputs,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )
    
    
    return response[len(prompt):] # TRUNCATE USER PROMPT
    

def summarize_with_llm(
    question,
    tool_results,
    model,
    tokenizer,
    max_new_tokens=512,
):
    """
    Convert structured tool outputs into a natural-language answer.
    """

    tool_results = json.dumps(
        tool_results,
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""You are a helpful biology assistant.
            A user asked a question, and reliable tools were used to gather information to answer it. 
            Your task is to summarize the information and provide a clear, 
            concise answer to the user's question.

            <question>
            {question}
            </question>

            <results>
            {tool_results}
            </results>

            
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return answer.strip()

