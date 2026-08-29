import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_ENCODER = SentenceTransformer(
    "BAAI/bge-m3",
    device=device,
)
print(TEXT_ENCODER)

texts = ["hi, this is a sample sentence"]

embeddings = TEXT_ENCODER.encode(
    texts,
    normalize_embeddings=False,
    batch_size=8,
)

print("shape:", embeddings.shape)
print(embeddings[0])