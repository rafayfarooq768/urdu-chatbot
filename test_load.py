import torch
from model import Transformer
import pickle

# load tokenizer to get vocab_size
tk = pickle.load(open("tokenizer.pkl","rb"))
vocab = tk["vocab"]
vocab_size = len(vocab)

# instantiate EXACTLY as during training:
embed_dim = 128
num_heads = 4
ff_hidden_dim = 512
num_layers = 2
max_len = 50

model = Transformer(vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_len)
state = torch.load("transformer_urdu_model.pth", map_location="cpu")

# quick parameter key + shape print
print("Checkpoint parameter keys and shapes (first 30):")
for k, v in list(state.items())[:30]:
    print(k, v.shape)

# load strict=True to ensure exact match
missing, unexpected = model.load_state_dict(state, strict=False)  # first try non-strict to see differences
print("Non-strict load done. Missing keys:", len([k for k in missing if missing]), " unexpected:", len(unexpected))
# Better: show mismatches explicitly by comparing shapes:
for name, param in model.state_dict().items():
    if name in state:
        if state[name].shape != param.shape:
            print("MISMATCH", name, " ckpt:", state[name].shape, " model:", param.shape)

# If all shapes match, load strict=True:
try:
    model.load_state_dict(state, strict=True)
    print("Loaded successfully with strict=True.")
except Exception as e:
    print("Strict load failed:", e)
