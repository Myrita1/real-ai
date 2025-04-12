import torch

# Simple whitespace tokenizer as placeholder
vocab = {word: i+1 for i, word in enumerate("this is a basic tokenizer example".split())}
vocab["[PAD]"] = 0
reverse_vocab = {i: word for word, i in vocab.items()}

def tokenize_input(text, seq_len=128):
    tokens = text.lower().split()
    ids = [vocab.get(t, 0) for t in tokens]
    padded = ids + [0] * (seq_len - len(ids))
    return torch.tensor([padded])

def decode_output(token_ids):
    decoded = [reverse_vocab.get(i.item(), "") for i in token_ids[0]]
    return " ".join(decoded).strip()