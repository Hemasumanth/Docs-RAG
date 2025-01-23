from pathlib import Path

WORK_DIR = Path(__file__).parent.parent

# Embedding dimensions
EMBEDDING_DIMENSIONS = {
    "thenlper/gte-base": 768,
    "thenlper/gte-large": 1024,
    "text-embedding-3-large": 3072,
    "Salesforce/SFR-Embedding-Mistral" : 4096
}

# Maximum context lengths
MAX_CONTEXT_LENGTHS = {
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096,
    "gpt-4-turbo-2024-04-09": 4096,
    "meta-llama/Llama-2-70b-chat-hf": 4096,
    "mistralai/Mistral-7B-Instruct-v0.1": 8192,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768
}