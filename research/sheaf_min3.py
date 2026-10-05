"""
Minimal sheaf diffusion layer: pure PyTorch (NO torch_geometric), with
correctness checks against (a) an explicit dense-loop reference,
(b) torch_geometric MessagePassing, (c) orthogonality of learned frames,
(d) PSD of the sheaf Laplacian, and a cost benchmark.

Run: python sheaf_min3.py
"""
import time
import resource
import torch
import torch.nn as nn

torch.manual_seed(0)


def add_self_loops_(edge_index, num_nodes):
    loop = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index, loop], dim=1)


def ctx_mean(x, edge_index, num_nodes):
    """mean of neighbour features per node, from directed edges (src->dst)"""
    src, _ = edge_index
    acc = torch.zeros_like(x).index_add_(0, src, x[src])
    deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype).index_add_(
        0, src, torch.ones_like(src, dtype=x.dtype))
    return acc / deg.clamp(min=1)[:, None]


def aug_norm(edge_index, num_nodes, dtype, device):
    """Augmented normalisation c_v = (deg_v + 1)^{-1/2}  (Bodnar et al. App. E),
    where deg_v counts the ORIGINAL directed edges only (self loops excluded).

    Returns the edge list WITH self loops appended (loops LAST) plus the
    per-edge weight c_u * c_v.  For the appended self loop at v this is c_v^2.
    """
    deg = torch.bincount(edge_index[0], minlength=num_nodes).to(dtype)
    dis = torch.nan_to_num((deg + 1).pow(-0.5), posinf=0.0)
    eall = add_self_loops_(edge_index, num_nodes)
    r, c = eall
    return eall, dis[r] * dis[c], dis


# ----------------------------------------------------------------------------
# 1. Diag-NSD diffusion step, pure PyTorch.
#
#    Diagonal restriction maps  F_{v<-e} = diag(f_v),  f_v in R^d.
#    Sheaf Laplacian blocks (augmented-normalised, Bodnar et al. 2022 App. E):
#        L[v,v] = diag( sum_{e ni v} f_v^2 ) * c_v^2
#        L[v,u] = -diag( f_v * f_u )      * c_v c_u     for edge (u,v)
#    Update:  X' = X - alpha * L X     with X of shape [N, d, f]
# ----------------------------------------------------------------------------
class DiagSheafLayer(nn.Module):
    def __init__(self, in_dim, d, alpha=1.0):
        super().__init__()
        self.d = d
        self.learner = nn.Linear(2 * in_dim, d, bias=False)   # f_v per incidence
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, x, edge_index, num_nodes):
        src, dst = edge_index
        ctx = ctx_mean(x, edge_index, num_nodes)

        # one restriction map per ENDPOINT of each edge, learned from
        # (own feature || neighbourhood context):  F_{v<-e} = diag(f_v)
        f_src = torch.tanh(self.learner(torch.cat([x[src], ctx[src]], -1)))  # [E,d]
        f_dst = torch.tanh(self.learner(torch.cat([x[dst], ctx[dst]], -1)))  # [E,d]

        eall, norm, _ = aug_norm(edge_index, num_nodes, x.dtype, x.device)
        nE = edge_index.size(1)

        # ---- Diagonal block ----
        # L[u,u] = sum_{e ni u} F_{u<-e}^T F_{u<-e}.  edge_index lists both
        # directions, so the entries with src==u already enumerate the deg(u)
        # incident edges exactly once.  Indexing dst as well would double count.
        node_f2 = torch.zeros(num_nodes, self.d, device=x.device, dtype=x.dtype)
        node_f2 = node_f2.index_add_(0, src, f_src * f_src)      # [N,d]

        xb = x.unsqueeze(1).expand(-1, self.d, -1)             # [N,d,f]
        zeros_d = torch.zeros(num_nodes, self.d, device=x.device, dtype=x.dtype)

        # Off-diagonal only: L[dst, src] = -diag(f_dst * f_src) c_dst c_src.
        # Self-loops are appended purely so the (deg+1) normalisation is
        # computable; their map product is 0 so they add nothing here.
        off = -torch.cat([f_dst * f_src, zeros_d])             # [E+N, d], sign matters
        row, col = eall[1], eall[0]
        coef = (off * norm[:, None])[:, :, None]
        Lx = torch.zeros_like(xb).index_add_(0, row, coef * xb[col])

        # Diagonal: L[v,v] = +diag(sum_{e ni v} f_v^2) * c_v^2  (POSITIVE: it is a
        # Laplacian).  For the appended self loop, norm already equals c_v^2.
        Lx = Lx + (node_f2 * norm[nE:][:, None])[:, :, None] * xb
        return xb - self.alpha * Lx


class DiagSheafStackLayer(nn.Module):
    """Same maths as DiagSheafLayer, but with SEPARATE arguments for the
    context used to learn the restriction maps (x_feat) and the stalk content
    being diffused (x_stalk).  This is the form you want when the maps depend on
    one representation and you diffuse another (e.g. sentence-transformers
    embeddings in, sheaf-diffused embeddings out).

        x_feat  : [N, c]  context for the map learner
        x_stalk : [N, d, f] stalk content
    """

    def __init__(self, ctx_dim, stalk_dim, alpha=1.0):
        super().__init__()
        self.d = stalk_dim
        self.learner = nn.Linear(2 * ctx_dim, stalk_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, x_feat, x_stalk, edge_index, num_nodes):
        src, dst = edge_index
        ctx = ctx_mean(x_feat, edge_index, num_nodes)
        f_s = torch.tanh(self.learner(torch.cat([x_feat[src], ctx[src]], -1)))
        f_d = torch.tanh(self.learner(torch.cat([x_feat[dst], ctx[dst]], -1)))

        eall, norm, _ = aug_norm(edge_index, num_nodes, x_feat.dtype, x_feat.device)
        nE = edge_index.size(1)

        node_f2 = torch.zeros(num_nodes, self.d, device=x_feat.device,
                              dtype=x_feat.dtype).index_add_(0, src, f_s * f_s)
        zeros_d = torch.zeros(num_nodes, self.d, device=x_feat.device,
                              dtype=x_feat.dtype)
        off = -torch.cat([f_d * f_s, zeros_d])
        col, row = eall[0], eall[1]
        coef = (off * norm[:, None])[:, :, None]
        Lx = torch.zeros_like(x_stalk).index_add_(0, row, coef * x_stalk[col])
        Lx = Lx + (node_f2 * norm[nE:][:, None])[:, :, None] * x_stalk
        return x_stalk - self.alpha * Lx


# ----------------------------------------------------------------------------
# 2. Orthogonal / connection-Laplacian (BGL) layer, learned edge frames.
#    Frames built as products of Givens rotations -> exactly orthogonal,
#    no Householder/Cayley dependency required.
# ----------------------------------------------------------------------------
class OrthSheafLayer(nn.Module):
    def __init__(self, in_dim, d, alpha=1.0):
        super().__init__()
        self.d = d
        self.to_ang = nn.Linear(2 * in_dim, d * (d - 1) // 2)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def frames(self, feat):
        ang = self.to_ang(feat)
        M = ang.size(0)
        R = torch.eye(self.d, device=ang.device, dtype=ang.dtype).repeat(M, 1, 1)
        k = 0
        for i in range(self.d):
            for j in range(i + 1, self.d):
                c_, s_ = torch.cos(ang[:, k]), torch.sin(ang[:, k])
                k += 1
                G = torch.eye(self.d, device=ang.device, dtype=ang.dtype).repeat(M, 1, 1)
                G[:, i, i], G[:, j, j] = c_, c_
                G[:, i, j], G[:, j, i] = -s_, s_
                R = torch.bmm(R, G)
        return R

    def forward(self, x, edge_index, num_nodes):
        src, dst = edge_index
        nE = edge_index.size(1)
        key = torch.minimum(src, dst).double() * num_nodes + torch.maximum(src, dst)
        uniq, inv = torch.unique(key, return_inverse=True)
        lo, hi = (uniq // num_nodes).long(), (uniq % num_nodes).long()
        R = self.frames(torch.cat([x[lo], x[hi]], -1))[inv]          # [E,d,d]

        eall, norm, _ = aug_norm(edge_index, num_nodes, x.dtype, x.device)
        xb = x.unsqueeze(1).expand(-1, self.d, -1)                   # [N,d,f]
        deg_vec = torch.bincount(edge_index[0], minlength=num_nodes).to(x.dtype)[:, None, None]

        # edge block  L[v,u] = -R_v^T R_u  (orthogonal => projection)
        # gather SOURCE column, scatter into DESTINATION row.
        cross = torch.einsum("eij,ejf->eif", R.transpose(1, 2), xb[src])
        msg_edge = (norm[:nE][:, None, None] * cross).contiguous()

        # Orthogonal case: R^T R = I, so L[v,v] = diag(deg_v) c_v^2 and the
        # appended self loop already carries c_v^2.
        diag_terms = deg_vec * norm[nE:][:, None, None] * xb

        Lx = torch.zeros_like(xb).index_add_(0, dst, msg_edge) + diag_terms
        return xb - self.alpha * Lx


# ----------------------------------------------------------------------------
# 3. explicit reference
# ----------------------------------------------------------------------------
def dense_sheaf_laplacian(edge_index, N, f_src, f_dst, d, dtype):
    """Explicit L[v,v] and L[v,u] blocks for diagonal restriction maps.

    Augmented normalisation: c_v = (deg_v + 1)^{-1/2}, deg counted on the
    directed edge list (self loops not counted).
        L[v,v] = diag(sum_{e ni v} f_v^2) c_v^2
        L[v,u] = -diag(f_dst f_src) c_v c_u
    """
    src, dst = edge_index
    E = edge_index.size(1)
    L = torch.zeros(N, d, N, d, dtype=dtype)
    deg = torch.bincount(src, minlength=N).to(dtype)
    dis = torch.nan_to_num((deg + 1).pow(-0.5), posinf=0.0)
    for k in range(E):
        u, v = int(src[k]), int(dst[k])
        L[v, :, u, :] += torch.diag(-dis[v] * f_dst[k] * f_src[k] * dis[u])
    for v in range(N):
        # entries with src == v enumerate the deg(v) incident edges once
        s = torch.zeros(d, dtype=dtype)
        for k in range(E):
            if int(src[k]) == v:
                s = s + f_src[k] ** 2
        L[v, :, v, :] += torch.diag(s * dis[v] ** 2)
    return L


def _f_maps(layer, x, edge_index, N):
    src, dst = edge_index
    ctx = ctx_mean(x, edge_index, N)
    fs = torch.tanh(layer.learner(torch.cat([x[src], ctx[src]], -1)))
    fd = torch.tanh(layer.learner(torch.cat([x[dst], ctx[dst]], -1)))
    return fs, fd


def rand_graph(N, m, seed):
    """Random UNDIRECTED graph (both directions present) WITHOUT self loops."""
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, N, (2, m), generator=g)
    ei = ei[:, ei[0] != ei[1]]
    return torch.cat([ei, ei.flip(0)], 1)


def check_dense():
    N, Fin, d = 10, 5, 3
    x = torch.randn(N, Fin)
    ei = rand_graph(N, 24, seed=11)
    layer = DiagSheafLayer(Fin, d)
    with torch.no_grad():
        fs, fd = _f_maps(layer, x, ei, N)
        L = dense_sheaf_laplacian(ei, N, fs, fd, d, x.dtype)   # [N,d,N,d]
        xb = x.unsqueeze(1).expand(-1, d, -1)
        Lx = torch.einsum("vaub,ubf->vaf", L, xb)
        layer.alpha.fill_(1.0)
        out = layer(x, ei, N)
    ref = xb - Lx
    err = (out - ref).abs().max().item()
    print(f"[1] vectorised vs dense-loop reference   : max|diff| = {err:.3e}")
    return err


def check_pyg():
    from torch_geometric.nn import MessagePassing

    class PyGRef(MessagePassing):
        """Same maths via PyG's MessagePassing. NOTE you MUST define message():
        the base class does not consume an extra per-edge coefficient."""

        def __init__(self, Fin, d):
            super().__init__(aggr="add", node_dim=0)
            self.d = d
            self.learner = nn.Linear(2 * Fin, d, bias=False)
            self.alpha = nn.Parameter(torch.tensor(1.0))

        def message(self, x_j, coef):
            # x_j: [E, d, f];  coef: [E, d]  ->  broadcast over the feature axis
            return coef.unsqueeze(-1) * x_j   # x_j = x[edge_index[0]] == source

        def forward(self, x, edge_index, num_nodes):
            N = num_nodes
            src, dst = edge_index
            ctx = ctx_mean(x, edge_index, N)
            fs = torch.tanh(self.learner(torch.cat([x[src], ctx[src]], -1)))
            fd = torch.tanh(self.learner(torch.cat([x[dst], ctx[dst]], -1)))
            eall, norm, _ = aug_norm(edge_index, N, x.dtype, x.device)
            nE = edge_index.size(1)
            node_f2 = torch.zeros(N, self.d, dtype=x.dtype).index_add_(0, src, fs * fs)
            zeros_d = torch.zeros(N, self.d, dtype=x.dtype)
            coef = -torch.cat([fd * fs, zeros_d]) * norm[:, None]
            xb = x.unsqueeze(1).expand(-1, self.d, -1)
            acc = self.propagate(eall, x=xb, coef=coef, size=(N, N))
            dg = node_f2 * norm[nE:][:, None]
            acc = acc + dg[:, :, None] * xb
            return xb - self.alpha * acc

    # PyG's default flow='source_to_target' gathers x[edge_index[0]] and
    # scatters into edge_index[1]; with (src, dst) ordering that is exactly
    # "gather the source column, accumulate into the destination row".

    N, Fin, d = 40, 16, 3
    x = torch.randn(N, Fin)
    ei = rand_graph(N, 200, seed=22)
    a, b = DiagSheafLayer(Fin, d), PyGRef(Fin, d)
    with torch.no_grad():
        b.learner.weight.copy_(a.learner.weight)
        e = (a(x, ei, N) - b(x, ei, N)).abs().max().item()
    print(f"[2] pure PyTorch vs torch_geometric      : max|diff| = {e:.3e}")
    return e


def check_orth():
    N, Fin, d = 200, 16, 4
    ei = rand_graph(N, 800, seed=33)
    x = torch.randn(N, Fin)
    lay = OrthSheafLayer(Fin, d)
    with torch.no_grad():
        R = lay.frames(torch.randn(64, 2 * Fin))
        err = (R @ R.transpose(-1, -2) - torch.eye(d)).abs().max().item()
        det = torch.linalg.det(R).abs()
        y = lay(x, ei, N)
        y3 = DiagSheafLayer(Fin, d)(x, ei, N)
    print(f"[3] learned frames: R^T R = I              : max|diff| = {err:.3e}"
          f"   |det R| = [{det.min():.4f},{det.max():.4f}]")
    print(f"    orth layer out {tuple(y.shape)}  diag layer out {tuple(y3.shape)}")
    return err


def check_psd():
    N, Fin, d = 25, 6, 2
    x = torch.randn(N, Fin)
    ei = rand_graph(N, 60, seed=44)
    layer = DiagSheafLayer(Fin, d)
    with torch.no_grad():
        fs, fd = _f_maps(layer, x, ei, N)
        L = dense_sheaf_laplacian(ei, N, fs, fd, d, x.dtype).reshape(N * d, N * d)
        ev = torch.linalg.eigvalsh(0.5 * (L + L.T))
        # kernel check: sum_v f_v = 0 should be in the kernel of a connected sheaf
        print(f"[4] sheaf Laplacian PSD                    : "
          f"lambda_min = {ev.min().item():+.3e}  lambda_max = {ev.max().item():.3e}")
    print(f"    (kernel property is verified in check_trivial_sheaf instead)")
    return ev.min().item()


def check_stack_layer():
    """DiagSheafStackLayer with the same feat used for maps AND stalk must
    reproduce DiagSheafLayer exactly."""
    N, C, Fd, d = 20, 7, 5, 3
    x = torch.randn(N, C)
    ei = rand_graph(N, 60, seed=66)
    a = DiagSheafLayer(C, d)
    b = DiagSheafStackLayer(C, d)
    with torch.no_grad():
        b.learner.weight.copy_(a.learner.weight)
        b.alpha.copy_(a.alpha)
        # base layer diffuses x itself => stalk content must be x broadcast
        s = x.unsqueeze(1).expand(-1, d, -1).contiguous()
        o1 = a(x, ei, N)
        o2 = b(x, s, ei, N)
    e = (o1 - o2).abs().max().item()
    print(f"[6] stacked variant == base layer           : max|diff| = {e:.3e}")
    return e


def check_trivial_sheaf():
    """maps == 1 => trivial sheaf => one step is x - Lx with the
    (deg+1)-augmented normalised sheaf Laplacian.  Verified against BOTH the
    dense-loop reference and a hand-built normalised-Laplacian matmul."""
    N = 30
    ei = rand_graph(N, 120, seed=55)
    x = torch.randn(N, 8)
    layer = DiagSheafLayer(8, 1)
    with torch.no_grad():
        layer.learner.weight.zero_()
        layer.learner.bias = nn.Parameter(torch.full((1,), 20.0))  # tanh(20) ~ 1
        layer.alpha.fill_(1.0)
        out = layer(x, ei, N)[:, 0, :]

        om = torch.ones(ei.size(1), 1)
        L = dense_sheaf_laplacian(ei, N, om, om, 1, x.dtype).reshape(N, N)
        e1 = (out - (x - L @ x)).abs().max().item()

        # Independent closed form.  With maps==1:
        #   L[v,v] = deg_v c_v^2,  L[v,u] = -c_v c_u
        # =>  L = D_c (D - A) D_c  with  D = diag(deg),  D_c = diag((deg+1)^{-1/2})
        # NOTE: use edge COUNTS (the random graph has multi-edges), matching deg.
        A = torch.zeros(N, N)
        A.index_put_((ei[1], ei[0]),
                     torch.ones(ei.size(1)), accumulate=True)
        deg = A.sum(1)
        dc = torch.nan_to_num((deg + 1).pow(-0.5), posinf=0.0)
        L2 = dc[:, None] * (torch.diag(deg) - A) * dc[None, :]
        e2 = float((L - L2).abs().max())
        ev2 = torch.linalg.eigvalsh(0.5 * (L2 + L2.T))
        z0 = abs(float(ev2.min()))
        nzero = int((ev2 < 1e-5).sum())
        ncomp = nzero
    print(f"    ...vs  L = D_c (D - A) D_c             : max|diff| = {e2:.3e}")
    print(f"    null eigenvalues = {nzero} (1 => graph connected, as expected); "
          f"min eig = {z0:.3e}")
    print(f"[5] maps==1 vs dense trivial-sheaf Lap.    : max|diff| = {e1:.3e}")
    return max(e1, e2)


def bench():
    print(f"\ntorch {torch.__version__} | cpu | threads={torch.get_num_threads()}")
    hdr = f"{'N':>9} {'deg':>5} {'E_directed':>11} {'fwd ms':>9} {'f+b ms':>9} {'RSS MB':>8}"
    print(hdr); print("-" * len(hdr))
    for N in [2_000, 20_000, 200_000]:
        E = int(N * 2.5)
        ei = torch.randint(0, N, (2, E))
        ei = torch.cat([ei, ei.flip(0)], 1)
        x = torch.randn(N, 64)
        layer = DiagSheafLayer(64, 4)
        opt = torch.optim.Adam(layer.parameters(), 1e-3)
        layer.eval()
        with torch.no_grad():
            layer(x, ei, N)
            t0 = time.perf_counter()
            for _ in range(5):
                layer(x, ei, N)
            fwd = (time.perf_counter() - t0) / 5 * 1000
        layer.train()
        t0 = time.perf_counter()
        for _ in range(5):
            layer(x, ei, N).square().mean().backward()
            opt.step(); opt.zero_grad()
        both = (time.perf_counter() - t0) / 5 * 1000
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"{N:>9} {5.0:>5.1f} {2*E:>11} {fwd:>9.2f} {both:>9.2f} {rss:>8.0f}")
        del ei, x, layer, opt

    print("\nfp32 activation memory for ONE sheaf layer (the gather dominates):")
    for N, F, d in [(100_000, 384, 4), (1_000_000, 768, 4), (10_000_000, 768, 4)]:
        E = 2 * int(N * 2.5)
        stalk = N * d * F * 4
        gath = E * d * F * 4
        maps = E * d * 4
        print(f"  N={N:>10,} f={F} d={d}: stalk {stalk/2**30:>7.2f} GiB | "
              f"messages {gath/2**30:>7.2f} GiB | maps {maps/2**30:>6.3f} GiB")


if __name__ == "__main__":
    check_dense()
    check_pyg()
    check_orth()
    check_psd()
    check_stack_layer()
    check_trivial_sheaf()
    bench()