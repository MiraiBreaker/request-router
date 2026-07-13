import streamlit as st
import os
import json
from llama_cpp import Llama
from openai import OpenAI

# Set up page styling
st.set_page_config(page_title="AI Agent Router Playground", page_icon="🤖", layout="wide")

st.title("🤖 Hybrid AI Agent Router")
st.markdown("""
This live demo showcases our **Two-Step Isolation Router**. 
1. **Local Gate (Gemma 4 2B):** Classifies the task as `SIMPLE` or `COMPLEX` within a strict 4GB RAM budget.
2. **Dynamic Divergence:** Simple tasks resolve instantly for **0 token cost**. Complex tasks route to the premium **Fireworks AI Cloud API**.
""")

# --- Sidebar Configuration ---
st.sidebar.header("🛠️ Infrastructure Settings")

# Handle API credentials securely via inputs or Streamlit Secrets
api_key = st.sidebar.text_input("Fireworks API Key", type="password", value=os.environ.get("FIREWORKS_API_KEY", ""))
base_url = st.sidebar.text_input("Fireworks Base URL", value="https://api.fireworks.ai/inference/v1")
selected_model = st.sidebar.selectbox("Cloud Model Target", [
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "accounts/fireworks/models/llama-v3p1-70b-instruct"
])

# --- Load Local Model Lazy-Style ---
@st.cache_resource
def load_local_gate():
    model_path = "models/gemma-4-E2B-it-Q4_K_M.gguf"
    # Fallback to download if it doesn't exist locally (useful for cloud deployments)
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        with st.spinner("Downloading Gemma 4 Router to application container..."):
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="lmstudio-community/gemma-4-E2B-it-GGUF",
                filename="gemma-4-E2B-it-Q4_K_M.gguf",
                local_dir="models"
            )
    return Llama(model_path=model_path, n_ctx=1024, n_threads=2, verbose=False)

try:
    local_llm = load_local_gate()
    st.sidebar.success("✅ Gemma 4 Router Ready (Local Memory)")
except Exception as e:
    st.sidebar.error(f"❌ Local Engine Error: {e}")
    local_llm = None

# --- Helper Functions ---
def classify_prompt(prompt):
    routing_instruction = (
        "You are a routing classification gate. Your ONLY job is to categorize the user query text.\n"
        "Output 'COMPLEX' if the query involves multi-step mathematics, writing code functions, debugging broken programs, or solving logic riddles/puzzles.\n"
        "Output 'SIMPLE' if the query is a simple sentiment analysis, short summary request, or extracting entity names.\n"
        "CRITICAL: You must answer with exactly one word: either COMPLEX or SIMPLE. Do not execute or answer the user prompt request itself."
    )
    formatted_prompt = f"<|im_start|>system\n{routing_instruction}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    output = local_llm(formatted_prompt, max_tokens=10, temperature=0.0, stop=["<|im_end|>", "<|im_start|>"])
    decision = output['choices'][0]['text'].strip().upper()
    return "SIMPLE" if "SIMPLE" in decision else "COMPLEX"

def execute_local(prompt):
    solver_instruction = "You are a concise execution agent. Answer the user prompt directly, cleanly, and briefly."
    formatted_prompt = f"<|im_start|>system\n{solver_instruction}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    output = local_llm(formatted_prompt, max_tokens=200, temperature=0.2, stop=["<|im_end|>", "<|im_start|>"])
    return output['choices'][0]['text'].split("<|im_start|>")[0].strip()

def execute_cloud(prompt):
    if not api_key:
        return "⚠️ [Simulated Mode]: Cloud route triggered! Add your Fireworks API Key in the sidebar to run live cloud execution."
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "Be concise. Provide only the direct answer to save tokens."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Cloud execution error: {str(e)}"

# --- Main UI Layout ---
user_input = st.text_area("✍️ Test your Prompt:", placeholder="e.g., Is this battery life good or bad? OR Write a python function to compute primes...")

if st.button("Route & Process Task", type="primary"):
    if user_input.strip() == "":
        st.warning("Please type a valid prompt first!")
    else:
        # Step 1: Classification Routing
        with st.spinner("Gemma 4 analyzing workload intent..."):
            route = classify_prompt(user_input)
        
        # UI Visual Callouts based on decision
        if route == "SIMPLE":
            st.metric(label="Decision Routing Metric", value="🟢 LOCAL TIER", delta="Cost: 0 Tokens Saved")
            with st.spinner("Processing locally on Gemma 4..."):
                answer = execute_local(user_input)
            st.success("### Final Output (Local Solver)")
            st.write(answer)
        else:
            st.metric(label="Decision Routing Metric", value="🔥 CLOUD TIER", delta="Cost: Premium API Request", delta_color="inverse")
            with st.spinner("Routing payload to Fireworks API cluster..."):
                answer = execute_cloud(user_input)
            st.info("### Final Output (Cloud API Cluster)")
            st.write(answer)
