"""
A transformer encoder block in plain NumPy, written to be read.

It runs a forward pass with random (untrained) weights so you can follow the
shapes and the maths of every step. It is a teaching model, not a trained one.

    python tiny_transformer.py
"""
import pathlib
import numpy as np

rng = np.random.default_rng(0)


class Config:
    d_model = 16      # size of every token vector
    n_heads = 4       # attention heads (d_model must divide by n_heads)
    d_ff = 64         # hidden size of the feed-forward layer
    n_layers = 2      # number of stacked blocks
    causal = False    # True = decoder style (GPT): a token cannot look at later tokens


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + eps)


def positional_encoding(n, d):
    """Sinusoidal positions: attention has no sense of order without this."""
    pos = np.arange(n)[:, None]
    i = np.arange(0, d, 2)[None, :]
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(pos / 10000 ** (i / d))
    pe[:, 1::2] = np.cos(pos / 10000 ** (i / d))
    return pe


class Block:
    def __init__(self, c):
        self.c = c
        s = 1 / np.sqrt(c.d_model)
        self.Wq, self.Wk, self.Wv, self.Wo = (rng.normal(0, s, (c.d_model, c.d_model)) for _ in range(4))
        self.W1 = rng.normal(0, s, (c.d_model, c.d_ff)); self.b1 = np.zeros(c.d_ff)
        self.W2 = rng.normal(0, 1 / np.sqrt(c.d_ff), (c.d_ff, c.d_model)); self.b2 = np.zeros(c.d_model)

    def attention(self, x):
        c, n = self.c, x.shape[0]
        dh = c.d_model // c.n_heads
        split = lambda m: m.reshape(n, c.n_heads, dh).transpose(1, 0, 2)      # (heads, n, dh)
        Q, K, V = split(x @ self.Wq), split(x @ self.Wk), split(x @ self.Wv)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)                       # (heads, n, n)
        if c.causal:
            scores = np.where(np.tril(np.ones((n, n))) == 1, scores, -1e9)
        weights = softmax(scores)                                             # each row sums to 1
        out = (weights @ V).transpose(1, 0, 2).reshape(n, c.d_model)          # join the heads
        return out @ self.Wo, weights

    def forward(self, x):
        a, w = self.attention(x)
        x = layer_norm(x + a)                                   # add and norm
        ff = np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2
        return layer_norm(x + ff), w                            # add and norm


def run(sentence="the cat sat on the mat", c=Config):
    tokens = sentence.split()
    vocab = {t: i for i, t in enumerate(sorted(set(tokens)))}
    ids = np.array([vocab[t] for t in tokens])
    E = rng.normal(0, 1, (len(vocab), c.d_model))               # embedding table
    x = E[ids] + positional_encoding(len(ids), c.d_model)
    print(f"tokens            : {tokens}")
    print(f"token ids         : {ids.tolist()}")
    print(f"after embedding   : {x.shape}  (tokens, d_model)")
    attn = None
    for i in range(c.n_layers):
        x, attn = Block(c).forward(x)
        print(f"after block {i + 1}     : {x.shape}  attention weights {attn.shape} (heads, tokens, tokens)")
    logits = x @ E.T                                            # score every vocabulary word
    probs = softmax(logits[-1])
    print(f"output logits     : {logits.shape}  (tokens, vocab)")
    print("next-token probabilities from the last position (untrained: it mostly echoes the last token, which is why training is needed):")
    for t, i in vocab.items():
        print(f"   {t:<4} {probs[i]:.3f}")
    return tokens, attn


def lora_demo(d=768, r=8):
    """LoRA: freeze W, train two small matrices A and B, use W + B @ A."""
    W = rng.normal(0, 0.02, (d, d))
    A = rng.normal(0, 0.01, (r, d)); B = np.zeros((d, r))       # B starts at zero, so nothing changes at step 0
    x = rng.normal(0, 1, d)
    same = np.allclose(W @ x, (W + B @ A) @ x)
    print(f"\nLoRA on one {d}x{d} weight matrix, rank {r}:")
    print(f"   full fine-tuning trains {W.size:,} numbers")
    print(f"   LoRA trains           {A.size + B.size:,} numbers ({100 * (A.size + B.size) / W.size:.1f}%)")
    print(f"   output unchanged before training: {same}")


if __name__ == "__main__":
    tokens, attn = run()
    lora_demo()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = pathlib.Path(__file__).parent / "results"; out.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, attn.shape[0], figsize=(4 * attn.shape[0], 4))
        for h, ax in enumerate(axes):
            ax.imshow(attn[h], cmap="viridis", vmin=0, vmax=attn.max())
            ax.set_xticks(range(len(tokens)), tokens, rotation=45); ax.set_yticks(range(len(tokens)), tokens)
            ax.set_title(f"Head {h + 1}")
        fig.suptitle("Attention weights in the last block (row attends to column, untrained weights)")
        fig.tight_layout(); fig.savefig(out / "attention_heads.png", dpi=140)
    except ImportError:
        pass
