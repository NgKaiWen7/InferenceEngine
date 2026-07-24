import re
import json

class BPETokenizer:
    def __init__(self, vocab_path, merges_path):
        self.vocab = self.load_vocab(vocab_path)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        self.bpe_ranks = self.load_merges(merges_path)
        self.cache = {}

        self.pat = re.compile(r"\S+|\s+")

    def load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_merges(self, path):
        merges = {}
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split()
                merges[(a, b)] = i
        return merges

    def get_pairs(self, word):
        pairs = set()
        prev = word[0]
        for curr in word[1:]:
            pairs.add((prev, curr))
            prev = curr
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = list(token)

        while True:
            pairs = self.get_pairs(word)

            # find best merge
            best_pair = None
            best_rank = float("inf")

            for p in pairs:
                rank = self.bpe_ranks.get(p)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = p

            if best_pair is None:
                break

            a, b = best_pair

            new_word = []
            i = 0

            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(a + b)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word

            if len(word) == 1:
                break

        self.cache[token] = word
        return word

    def encode(self, text):
        tokens = []

        for chunk in self.pat.findall(text):
            chunk = chunk.strip()
            if not chunk:
                continue

            pieces = self.bpe(chunk)

            for p in pieces:
                tokens.append(self.vocab.get(p, self.vocab.get("<unk>", 0)))

        return tokens

    def decode(self, ids):
        return "".join(self.id_to_token[i] for i in ids)

# tokenizer = BPETokenizer("models/qwen3-4B/vocab.json", "models/qwen3-4B/merges.txt")
# encoded = tokenizer.encode("mello")
# decoded = tokenizer.decode(encoded)
# print(encoded)
# print(decoded)


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

token_id = 14990


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="Qwen/Qwen3-4B")

inputs = tokenizer("Hello", return_tensors="pt")
layer = model.model.layers[0]

print(layer.self_attn.q_proj.weight.shape)
print(layer.self_attn.k_proj.weight.shape)
print(layer.self_attn.v_proj.weight.shape)
print(layer.self_attn.o_proj.weight.shape)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1000)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))