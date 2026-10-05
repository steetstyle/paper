"""
Recipe: sentence-transformers embeddings -> sheaf diffusion -> retrieval.
Self-contained; no network access needed (uses a deterministic hash embedding).

Run: python sentence_sheaf.py
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# reuse the verified layer from sheaf_min3.py
# ---------------------------------------------------------------------------
import importlib.util
spec = importlib.util.spec_from_file_location("sm", str(__import__('pathlib').Path(__file__).with_name('sheaf_min3.py')))
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)


# ---------------------------------------------------------------------------
# 1. Stand-in for sentence-transformers: [N, 384] dense embeddings.
#    Replace with: SentenceTransformer("all-MiniLM-L6-v2").encode(texts)
# ---------------------------------------------------------------------------
def fake_embeddings(N, dim=384, n_clusters=12, seed=42):
    """Stand-in for a sentence-transformer encoder: returns N L2-normalised
    vectors that already carry cluster structure (like real embeddings do)."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, n_clusters, (N,), generator=g)
    centers = F.normalize(torch.randn(n_clusters, dim, generator=g), dim=-1)
    x = centers[labels] * 0.9 + 0.1 * F.normalize(torch.randn(N, dim, generator=g), dim=-1)
    return F.normalize(x, dim=-1), labels


# ---------------------------------------------------------------------------
# 2. Build the document graph from metadata (no model needed)
# ---------------------------------------------------------------------------
def build_graph(labels, n_clusters=12, device="cpu"):
    """Ring-within-cluster graph. labels: [N]"""
    N = labels.numel()
    g = torch.Generator().manual_seed(7)
    grps = [torch.nonzero(labels == c).flatten() for c in range(n_clusters)]
    src, dst = [], []
    for grp in grps:
        k = grp.numel()
        if k < 2:
            continue
        perm = grp[torch.randperm(k, generator=g)]
        nxt = torch.roll(perm, -1)
        src += perm.tolist(); dst += nxt.tolist()
    ei = torch.tensor([src, dst], dtype=torch.long)
    ei = torch.cat([ei, ei.flip(0)], dim=1)          # symmetrise
    return ei.to(device), labels.to(device)


# ---------------------------------------------------------------------------
# 3. Model: encoder -> sheaf layers -> readout -> projection for ANN
# ---------------------------------------------------------------------------
class SheafRetrieval(nn.Module):
    def __init__(self, in_dim, emb_dim=384, stalk=4, hid=64, n_layers=2, alpha=1.0):
        super().__init__()
        self.stalk = stalk
        self.in_dim = in_dim
        self.hid = hid
        self.enc = nn.Linear(in_dim, hid, bias=False)      # context for the map learner
        self.feat = nn.Linear(in_dim, hid, bias=False)     # stalk content init
        self.layers = nn.ModuleList([
            sm.DiagSheafStackLayer(hid, stalk, alpha=alpha) for _ in range(n_layers)
        ])
        self.readout = nn.Linear(stalk * hid, emb_dim, bias=False)

    def forward(self, x, edge_index, N):
        # keep x (the sentence-transformer embedding) as the map context
        x_feat = F.elu(self.enc(x))                        # [N, hid]
        # stalk content: one channel per stalk dimension, initialised from x
        # project the incoming embedding to the per-stalk-channel width
        s = self.feat(x).unsqueeze(1).expand(-1, self.stalk, -1).contiguous()
        for lyr in self.layers:
            s = F.elu(lyr(x_feat, s, edge_index, N))
        return F.normalize(self.readout(s.reshape(s.size(0), -1)), dim=-1)


# ---------------------------------------------------------------------------
# 4. Loss: standard contrastive / InfoNCE over a graph-aware batch.
#    NO torch_geometric batching needed - transductive full-graph passes only.
# ---------------------------------------------------------------------------
def infonce(q, pos, temperature=0.05):
    logits = q @ pos.t() / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


def train(N=20_000, epochs=30, device="cpu"):
    x, labels = fake_embeddings(N)
    x = x.to(device)
    ei, labels = build_graph(labels, device=device)
    model = SheafRetrieval(x.size(1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

    # anchor/positive pairs = edges (doc cites a related doc)
    # anchor/positive pairs = edges (doc links to a related doc).
    # Sample a SMALL batch: InfoNCE is |B| x |B| logits, so keep |B| ~ 2k.
    src, dst = ei[0], ei[1]
    B = 2048

    t0 = time.perf_counter()
    for ep in range(epochs):
        z = model(x, ei, N)                     # full-graph pass (transductive)
        k = min(B, src.numel())
        perm = torch.randint(0, src.numel(), (k,), device=src.device)
        loss = infonce(z[src[perm]], z[dst[perm]])
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:>3}  loss {loss.item():.4f}  "
                  f"({time.perf_counter()-t0:5.1f}s)")
    return model, x, ei, labels


@torch.no_grad()
def evaluate(z, labels):
    """Mean intra-cluster cosine (higher = better structure retained)."""
    zn = F.normalize(z, dim=-1)
    # sample 4k nodes to avoid the N x N matrix at scale
    n = labels.numel()
    idx = torch.randperm(n, device=labels.device)[:4000]
    zz, ll = zn[idx], labels[idx]
    sim = zz @ zz.t()
    same = ll[:, None] == ll[None, :]
    eye = torch.eye(len(idx), dtype=torch.bool, device=z.device)
    m = same & ~eye
    return float(sim[m].mean())


if __name__ == "__main__":
    N = 20_000
    x, labels = fake_embeddings(N)
    ei, labels = build_graph(labels)
    print(f"graph: N={N}, directed edges={ei.size(1)}, feat dim={x.size(1)}, "
          f"avg deg={ei.size(1)/N:.1f}")

    print("\n-- baseline: cosine on raw (sentence-transformer-like) embeddings --")
    print(f"   intra-cluster cosine = {evaluate(x, labels):.4f}")

    print("\n-- training sheaf model (30 epochs, CPU) --")
    t0 = time.perf_counter()
    model, x, ei, labels = train(N=N, epochs=30)
    train_s = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        z = model(x, ei, N)
    print(f"\n   trained in {train_s:.1f}s for 30 epochs "
          f"({train_s/30*1000:.0f} ms/epoch)")
    print(f"   intra-cluster cosine after sheaf diffusion = {evaluate(z, labels):.4f}")

    # retrieval: one sheaf forward = one pass over the WHOLE graph
    print("\n-- inference cost --")
    with torch.no_grad():
        model(x, ei, N)
        t0 = time.perf_counter()
        for _ in range(5):
            z = model(x, ei, N)
        full = (time.perf_counter() - t0) / 5 * 1000
    print(f"   full-graph diffusion pass: {full:.1f} ms for N={N} "
          f"({full/5:.1f} ms per single query - NOT query-scalable)")
    print(f"   scaling: this is O(E) per layer, per query. 1M nodes -> ~"
          f"{full*50/1000:.0f}s/query on this CPU.")