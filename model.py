# model.py
import torch
import torch.nn as n
import torch.nn.functional as F
import math

# ------------------------------
# Embedding + Positional Encoding
# ------------------------------
class tokenPositionAmbed(n.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(tokenPositionAmbed, self).__init__()
        self.embed = n.Embedding(vocab_size, embed_dim)

        # build positional encodings on CPU
        pos = torch.zeros(max_len, embed_dim)
        for p in range(max_len):
            for i in range(0, embed_dim, 2):
                pos[p, i] = math.sin(p / (10000 ** (i / embed_dim)))
                if i + 1 < embed_dim:
                    pos[p, i + 1] = math.cos(p / (10000 ** (i / embed_dim)))
        pos = pos.unsqueeze(0)  # shape [1, max_len, embed_dim]

        # register as buffer so it automatically moves with model.to(device)
        self.register_buffer('pos', pos, persistent=False)

    def forward(self, x):
        emb = self.embed(x)
        emb = emb + self.pos[:, :emb.size(1), :].to(emb.device)
        return emb

# ------------------------------
# Feed-forward network
# ------------------------------
class FeedForwardNetwork(n.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = n.Linear(embed_dim, hidden_dim)
        self.activation = n.ReLU()
        self.layer2 = n.Linear(hidden_dim, embed_dim)
        self.dropout = n.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x

# ------------------------------
# Scaled dot-product attention
# ------------------------------
class sdpAttention(n.Module):
    def __init__(self):
        super(sdpAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            # mask expected to be broadcastable, original code used mask==0 fill
            score = score.masked_fill(mask == 0, -1e9)
        aw = F.softmax(score, dim=-1)
        op = torch.matmul(aw, v)
        return op, aw

# ------------------------------
# Multi-head attention (custom)
# ------------------------------
class Mhattention(n.Module):
    def __init__(self, ed, nh):
        super(Mhattention, self).__init__()
        self.ed = ed
        self.nh = nh
        self.hd = ed // nh

        self.q = n.Linear(ed, ed)
        self.k = n.Linear(ed, ed)
        self.v = n.Linear(ed, ed)
        self.fc_out = n.Linear(ed, ed)
        self.att = sdpAttention()

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = q.view(batch_size, -1, self.nh, self.hd).transpose(1, 2)
        k = k.view(batch_size, -1, self.nh, self.hd).transpose(1, 2)
        v = v.view(batch_size, -1, self.nh, self.hd).transpose(1, 2)

        out, attn_weights = self.att(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.ed)
        out = self.fc_out(out)
        return out, attn_weights

# ------------------------------
# Encoder block
# ------------------------------
class encoder(n.Module):
    def __init__(self, ed, nh, ffhd, dropout=0.1):
        super(encoder, self).__init__()
        self.mha = Mhattention(ed, nh)
        self.ffn = FeedForwardNetwork(ed, ffhd, dropout)
        self.norm1 = n.LayerNorm(ed)
        self.norm2 = n.LayerNorm(ed)
        self.dropout = n.Dropout(dropout)

    def forward(self, x, mask=None):
        attout, _ = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attout))
        fout = self.ffn(x)
        x = self.norm2(x + self.dropout(fout))
        return x

# ------------------------------
# Decoder layer
# ------------------------------
class decode(n.Module):
    def __init__(self, ed, nh, ffhd, dropout=0.1):
        super(decode, self).__init__()
        self.selfAtt = Mhattention(ed, nh)
        self.encDecAtt = Mhattention(ed, nh)
        self.ffn = FeedForwardNetwork(ed, ffhd, dropout)

        self.norm1 = n.LayerNorm(ed)
        self.norm2 = n.LayerNorm(ed)
        self.norm3 = n.LayerNorm(ed)
        self.dropout = n.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, trg_mask=None):
        attn1, _ = self.selfAtt(x, x, x, trg_mask)
        x = self.norm1(x + self.dropout(attn1))

        attn2, _ = self.encDecAtt(x, enc_out, enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout(attn2))

        ffnOut = self.ffn(x)
        x = self.norm3(x + self.dropout(ffnOut))

        return x

# ------------------------------
# Decoder (stack)
# ------------------------------
class Decoder(n.Module):
    def __init__(self, vs, ed, ml, nl, nh, ffhd, dropout=0.1):
        super(Decoder, self).__init__()
        self.embd = n.Embedding(vs, ed)
        self.pe = torch.zeros(ml, ed)
        for p in range(ml):
            for i in range(0, ed, 2):
                self.pe[p, i] = math.sin(p / (10000 ** (i / ed)))
                if i + 1 < ed:
                    self.pe[p, i + 1] = math.cos(p / (10000 ** (i / ed)))
        self.pe = self.pe.unsqueeze(0)  # [1, max_len, ed]

        self.layers = n.ModuleList([decode(ed, nh, ffhd, dropout) for _ in range(nl)])
        self.fcout = n.Linear(ed, vs)
        self.dropout = n.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # move pos to device
        pe = self.pe.to(x.device)
        x = self.embd(x) + pe[:, :x.size(1), :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        out = self.fcout(x)
        return out

# ------------------------------
# Full Transformer (matching your training script)
# ------------------------------
class Transformer(n.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = tokenPositionAmbed(vocab_size, embed_dim, max_len)
        self.encoder_layers = n.ModuleList([
            encoder(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.decoder = Decoder(vocab_size, embed_dim, max_len, num_layers, num_heads, ff_hidden_dim, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder_embedding(src)
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        out = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return out

# export helper mask generator used in training notebook
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz)), diagonal=1).bool()
    return mask
