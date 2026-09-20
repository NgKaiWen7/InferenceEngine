from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
MODEL_NAME = "Qwen/Qwen3-1.7B"

MODEL_DIR = Path("./Qwen3-1.7B")
TOKENIZER_DIR = Path("./Qwen3-1.7B-tokenizer")


def load_model():
    if TOKENIZER_DIR.exists():
        print(f"Loading tokenizer from {TOKENIZER_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    else:
        print(f"Downloading tokenizer from {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(TOKENIZER_DIR)

    if MODEL_DIR.exists():
        print(f"Loading model from {MODEL_DIR}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    else:
        print(f"Downloading model from {MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.save_pretrained(MODEL_DIR)

    return tokenizer, model

tokenizer, model = load_model()

print(type(model))
input_ids = torch.tensor([0])
token_embeddings = model.model.embed_tokens(input_ids)
print(token_embeddings)
