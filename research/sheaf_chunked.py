"""
Fix for the [E,d,f] message-gather OOM: chunk the edge pass.
Shows the same math, same results, bounded memory.

Run: python sheaf_chunked.py
"""
import time
import torch
import torch.nn as nn
import importlib.util

spec = importlib.util.spec_from_file_location("sm", str(__import__('pathlib').Path(__file__).with_name('sheaf_min3.py')))
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)


class ChunkedDiagSheafLayer(sm.DiagSheafLayer):
    """Identical maths to DiagSheafLayer, but the [E,d,f] gather is done in
    edge chunks and accumulated, so peak memory is O(chunk*d*f) instead of
    O(E*d*f).  Mathematically identical (accumulation is a sum)."""

    def __init__(self, *a, chunk=1 << 18, **kw):
        super().__init__(*a, **kw)
        self.chunk = chunk

    def forward(self, x, edge_index, num_nodes):
        src, dst = edge_index
        nE = edge_index.size(1)
        ctx = sm.ctx_mean(x, edge_index, num_nodes)

        # map learning is only [E, d] -> cheap, do it whole
        f_s = torch.tanh(self.learner(torch.cat([x[src], ctx[src]], -1)))
        f_d = torch.tanh(self.learner(torch.cat([x[dst], ctx[dst]], -1)))
        node_f2 = torch.zeros(num_nodes, self.d, device=x.device,
                              dtype=x.dtype).index_add_(0, src, f_s * f_s)

        eall, norm, _ = sm.aug_norm(edge_index, num_nodes, x.dtype, x.device)
        zeros_d = torch.zeros(num_nodes, self.d, device=x.device, dtype=x.dtype)
        off = -torch.cat([f_d * f_s, zeros_d])          # [E+N, d]
        col, row = eall[0], eall[1]

        xb = x.unsqueeze(1).expand(-1, self.d, -1)
        Lx = torch.zeros_like(xb)
        C = self.chunk
        for lo in range(0, off.size(0), C):
            hi = min(lo + C, off.size(0))
            coef = (off[lo:hi] * norm[lo:hi, None])[:, :, None]
            Lx.index_add_(0, row[lo:hi], coef * xb[col[lo:hi]])

        Lx = Lx + (node_f2 * norm[nE:][:, None])[:, :, None] * xb
        return xb - self.alpha * Lx


def check_identical():
    N, C, d = 300, 16, 4
    x = torch.randn(N, C)
    ei = sm.rand_graph(N, 1200, seed=5)
    a = sm.DiagSheafLayer(C, d)
    b = ChunkedDiagSheafLayer(C, d, chunk=97)   # deliberately awkward chunk
    with torch.no_grad():
        b.learner.weight.copy_(a.learner.weight)
        b.alpha.copy_(a.alpha)
        e = (a(x, ei, N) - b(x, ei, N)).abs().max().item()
    print(f"[chunked == unchunked]  max|diff| = {e:.3e}  (chunk=97, ragged)")
    return e


def bench():
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("no CUDA; skipping GPU memory test")
        return
    print(f"\nGPU: {torch.cuda.get_device_name(0)}  "
          f"cap {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
    print(f"{'N':>10} {'E_dir':>10} {'f':>5} {'d':>3} {'chunk':>8} "
          f"{'ms':>8} {'peak GiB':>9}")
    for N, f, d, ch in [
        (200_000, 384, 4, 1 << 14),
        (200_000, 384, 4, 1 << 12),
        (1_000_000, 384, 4, 1 << 12),
        (2_000_000, 384, 4, 1 << 12),
    ]:
        try:
            E = int(N * 2.5)
            ei = torch.randint(0, N, (2, E), device=dev)
            ei = torch.cat([ei, ei.flip(0)], 1)
            x = F.normalize(torch.randn(N, f, device=dev), dim=-1)
            L = ChunkedDiagSheafLayer(f, d, chunk=ch).to(dev)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                L(x, ei, N)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(3):
                    L(x, ei, N)
                torch.cuda.synchronize()
                t = (time.perf_counter() - t0) / 3 * 1000
            print(f"{N:>10,} {ei.size(1):>10,} {f:>5} {d:>3} {ch:>8,} "
                  f"{t:>8.1f} {torch.cuda.max_memory_allocated()/2**30:>9.2f}")
            del ei, x, L
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"{N:>10,} {'-':>10} {f:>5} {d:>3} {ch:>8,} "
                  f"{'OOM':>8} {str(e)[:40]}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    check_identical()
    bench()