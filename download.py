from transformers import AutoTokenizer, AutoProcessor, CLIPModel
from huggingface_hub import snapshot_download

model_id = "trollathon/bge-m3-safetensors"
local_dir = "./bge-m3-safetensors"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print(f"Downloaded to: {local_dir}")


model_id = "FacebookAI/xlm-roberta-base"
local_dir = "./xlm-roberta-base"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print(f"Downloaded to: {local_dir}")