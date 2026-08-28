from transformers import pipeline


pipeline = pipeline(
    task="fill-mask",
    model="FacebookAI/xlm-roberta-base",
    device=0
)
# Example in French
output = pipeline("Bonjour, je suis un modèle <mask>.")
print(output)