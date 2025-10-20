import pickle

def load_tokenizer(path="tokenizer.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["vocab"], data["idx_to_word"]

def encode(sentence, vocab, max_len=20):
    import re
    sentence = re.sub(r'[^\w\s]', '', sentence)
    tokens = sentence.split()
    ids = [vocab.get("<SOS>", 1)]
    for t in tokens:
        ids.append(vocab.get(t, vocab.get("<UNK>", 3)))
    ids.append(vocab.get("<EOS>", 2))
    if len(ids) > max_len:
        ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(vocab.get("<PAD>", 0))
    return ids

def decode(token_ids, idx_to_word):
    words = [idx_to_word.get(i, "<UNK>") for i in token_ids]
    if "<EOS>" in words:
        words = words[:words.index("<EOS>")]
    return " ".join([w for w in words if w not in ("<PAD>", "<SOS>", "<EOS>")])
