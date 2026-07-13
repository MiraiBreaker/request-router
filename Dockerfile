FROM python:3.11-slim

# Install light build tools required to compile llama-cpp
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install standard dependencies
RUN pip install --no-cache-dir openai llama-cpp-python huggingface_hub

# Create model directory and pull Gemma-4-E2B-it cleanly inside the image
RUN mkdir -p models && \
    curl -L -o models/gemma-4-E2B-it-Q4_K_M.gguf \
    https://huggingface.co/lmstudio-community/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf

# Copy your optimized agent code
COPY agent.py .

# Create the evaluation platform fallback folders
RUN mkdir -p /input /output

# Command to execute on startup
CMD ["python", "agent.py"]
