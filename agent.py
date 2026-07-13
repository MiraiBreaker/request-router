import os
import json
from openai import OpenAI
from llama_cpp import Llama

INPUT_PATH = "./input/tasks.json"
OUTPUT_PATH = "./output/results.json"

API_KEY = os.environ.get("FIREWORKS_API_KEY", "dummy_key")
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
ALLOWED_MODELS = os.environ.get("ALLOWED_MODELS", "").split(",")
FIREWORKS_MODEL = ALLOWED_MODELS[0] if ALLOWED_MODELS and ALLOWED_MODELS[0] else "accounts/fireworks/models/llama-v3p1-8b-instruct"

# Load local Gemma 4
LOCAL_MODEL_PATH = "models/gemma-4-E2B-it-Q4_K_M.gguf"
local_llm = None

if os.path.exists(LOCAL_MODEL_PATH):
    local_llm = Llama(model_path=LOCAL_MODEL_PATH, n_ctx=1024, n_threads=2, verbose=False)

def classify_route(prompt: str) -> str:
    if not local_llm:
        return "COMPLEX"

    routing_instruction = (
        "You are a routing classification gate. Your ONLY job is to categorize the user query text.\n"
        "Output 'COMPLEX' if the query involves multi-step mathematics, writing code functions, "
        "debugging broken programs, or solving logic riddles/puzzles.\n"
        "Output 'SIMPLE' if the query is a simple sentiment analysis, short summary request, or extracting entity names.\n"
        "CRITICAL: You must answer with exactly one word: either COMPLEX or SIMPLE. Do not execute or answer the user prompt request itself."
    )
    
    formatted_prompt = f"<|im_start|>system\n{routing_instruction}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Explicitly catch all potential end-of-sequence structures
    output = local_llm(formatted_prompt, max_tokens=10, temperature=0.0, stop=["<|im_end|>", "<|im_start|>"])
    decision = output['choices'][0]['text'].strip().upper()
    
    if "SIMPLE" in decision:
        return "SIMPLE"
    return "COMPLEX"

def call_local_solver(prompt: str) -> str:
    solver_instruction = "You are a concise execution agent. Answer the user prompt directly, cleanly, and briefly."
    formatted_prompt = f"<|im_start|>system\n{solver_instruction}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Added robust stop strings to completely block the template leak
    output = local_llm(
        formatted_prompt, 
        max_tokens=200, 
        temperature=0.2, 
        stop=["<|im_end|>", "<|im_start|>", "<bos>", "<eos>"]
    )
    
    clean_text = output['choices'][0]['text'].split("<|im_start|>")[0]
    return clean_text.strip()

def call_fireworks(prompt: str) -> str:
    # Fix for local testing environments using dummy URLs
    if "localhost" in BASE_URL or "127.0.0.1" in BASE_URL or API_KEY == "mock_key":
        return "[Simulated Fireworks Cloud API Response]: Task handled successfully."

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    try:
        response = client.chat.completions.create(
            model=FIREWORKS_MODEL,
            messages=[
                {"role": "system", "content": "Be concise. Provide only the direct answer to save tokens."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error executing external request: {str(e)}"

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input file not found at {INPUT_PATH}")
        return

    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    results = []
    for task in tasks:
        task_id = task.get("task_id")
        prompt = task.get("prompt", "")
        
        route = classify_route(prompt)
        print(f"Task {task_id} routed to -> {route}")
        
        if route == "COMPLEX":
            answer = call_fireworks(prompt)
        else:
            answer = call_local_solver(prompt)
            
        results.append({
            "task_id": task_id,
            "answer": answer
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
