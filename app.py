import streamlit as st
import torch
import pickle
import re
from model import Transformer, generate_square_subsequent_mask

# ==============================
# Load Tokenizer
# ==============================
with open("tokenizer.pkl", "rb") as f:
    data = pickle.load(f)
vocab = data["vocab"]
idx_to_token = data.get("idx_to_token") or data.get("idx_to_word")
vocab_size = len(vocab)

# ==============================
# Load Model
# ==============================
device = torch.device("cpu")

model = Transformer(
    vocab_size=vocab_size,
    embed_dim=128,
    num_heads=4,
    ff_hidden_dim=512,
    num_layers=2,
    max_len=50
)

state = torch.load("transformer_urdu_model.pth", map_location=device)
model.load_state_dict(state, strict=True)
model.to(device)
model.eval()

# ==============================
# Helper Functions
# ==============================
def tokenize(text):
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def encode_sentence(sentence, vocab, max_len=50):
    tokens = tokenize(sentence)
    ids = [vocab.get("<SOS>", 1)]
    for t in tokens:
        ids.append(vocab.get(t, vocab.get("<UNK>", 3)))
    ids.append(vocab.get("<EOS>", 2))
    if len(ids) > max_len:
        ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(vocab.get("<PAD>", 0))
    return ids

def decode_ids(ids, idx_to_token):
    words = [idx_to_token.get(i, "<UNK>") for i in ids]
    words = [w for w in words if w not in ("<PAD>", "<SOS>", "<EOS>")]
    return " ".join(words)

@torch.no_grad()
def greedy_generate(model, src_ids, device, max_len=50, sos=1, eos=2):
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    memory = model.encoder_embedding(src_tensor)
    
    for layer in model.encoder_layers:
        memory = layer(memory)
    
    generated = [sos]
    for _ in range(max_len):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
        out = model.decoder(tgt, memory, tgt_mask=tgt_mask)
        logits = out[:, -1, :]
        next_tok = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_tok)
        if next_tok == eos:
            break
    return generated[1:]

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="🤖 Urdu Transformer Chatbot", layout="centered")

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #f8f8f8;
    font-family: "Segoe UI", "Noto Nastaliq Urdu", sans-serif;
}
.chatbox {
    max-width: 800px;
    margin: 20px auto;
    background-color: #0e1117;
    padding: 10px;
}
.message {
    border-radius: 8px;
    padding: 10px 14px;
    margin: 10px 0;
    max-width: 85%;
    word-wrap: break-word;
    line-height: 1.6;
    direction: rtl;
    font-size: 17px;
}
.user {
    background-color: #1e222a;
    color: #eaeaea;
    margin-left: auto;
    text-align: right;
}
.bot {
    background-color: #2a2f3a;
    color: #ffffff;
    margin-right: auto;
    text-align: left;
}
.stTextInput > div > div > input {
    border-radius: 8px;
    background-color: #1e222a;
    color: #f8f8f8;
    border: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 Urdu Transformer Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_input():
    user_input = st.session_state.msg_input.strip()
    if not user_input:
        return
    st.session_state.chat_history.append(("user", user_input))
    
    src_ids = encode_sentence(user_input, vocab)
    out_ids = greedy_generate(model, src_ids, device)
    bot_reply = decode_ids(out_ids, idx_to_token)
    st.session_state.chat_history.append(("bot", bot_reply))
    st.session_state.msg_input = ""

# ===== Chat Display =====
st.markdown("<div class='chatbox'>", unsafe_allow_html=True)
for role, msg in st.session_state.chat_history:
    css_class = "user" if role == "user" else "bot"
    st.markdown(f"<div class='message {css_class}'>{msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ===== Input Field =====
st.text_input("اپنا پیغام لکھیں:", key="msg_input", on_change=process_input, label_visibility="collapsed")
