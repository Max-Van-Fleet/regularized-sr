import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- classical steps (feature-space) ----------
def _grad_x(u): return u[..., :, 1:] - u[..., :, :-1]
def _grad_y(u): return u[..., 1:, :] - u[..., :-1, :]

def _div(px, py):
    """
    Divergence adjoint to forward differences:
      grad_x u -> px shape [B,C,H,W-1]
      grad_y u -> py shape [B,C,H-1,W]
    Neumann boundary. Returns [B,C,H,W].
    """
    divx = F.pad(px, (0, 1, 0, 0)) - F.pad(px, (1, 0, 0, 0))
    divy = F.pad(py, (0, 0, 0, 1)) - F.pad(py, (0, 0, 1, 0))
    return divx + divy

def tv_step(x, lam, step=0.25, eps=1e-6):
    gx = _grad_x(x)  # [B,C,H,W-1]
    gy = _grad_y(x)  # [B,C,H-1,W]
    qx = gx / (gx.abs() + eps)
    qy = gy / (gy.abs() + eps)
    d = _div(qx, qy)                               # [B,C,H,W]
    return x - step * lam * d                      # lam can be [B,C,1,1] or [B,C,H,W]

def anisotropic_diffusion_step(x, lam, step=0.25, kappa=0.03, eps=1e-6):
    gx = _grad_x(x); gy = _grad_y(x)
    g_x = 1.0 / (1.0 + (gx.abs() / kappa) ** 2)
    g_y = 1.0 / (1.0 + (gy.abs() / kappa) ** 2)
    qx = g_x * gx / (gx.abs() + eps)
    qy = g_y * gy / (gy.abs() + eps)
    d = _div(qx, qy)
    return x - step * lam * d

def laplacian_step(x, lam, step=0.25):
    B, C, H, W = x.shape
    k = torch.tensor([[0., 1., 0.],
                      [1., -4., 1.],
                      [0., 1., 0.]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    weight = k.repeat(C, 1, 1, 1)                  # depthwise
    lap = F.conv2d(x, weight, padding=1, groups=C)
    return x - step * lam * lap

# --------- mixer module ----------
class ChannelRegMixer(nn.Module):
    """
    For each channel, predict a mixture over K regularizers and apply T steps.
    Regularizers (K=3): [TV, Laplacian (Tikhonov), Anisotropic diffusion]
    Content-awareness comes from:
      - richer per-channel stats for the mixture head,
      - a depthwise spatial gate producing λ(x) over H×W.
    """
    def __init__(self, channels, steps=2, base_lambda=0.03, kappa=0.03,
                 hidden=0, temperature=1.0, use_gumbel=False,
                 gate_bias=-2.0, init_aniso_bias=0.2, init_strength=1.0):
        super().__init__()
        self.steps = steps
        self.base_lambda = base_lambda
        self.kappa = kappa
        self.temperature = temperature
        self.use_gumbel = use_gumbel

        # Gating head: per-channel mixture over K regs
        K = 3
        self.K = K

        # stats extractor: per-channel global descriptors
        # [B,C,H,W] -> [B,C,6]: GAP, GMP, mean|∇|, mean|Δ|, VAR, anisotropy(|gx|^2 vs |gy|^2)
        self.pool_gap = nn.AdaptiveAvgPool2d(1)
        self.pool_gmp = nn.AdaptiveMaxPool2d(1)

        in_feats = 6
        if hidden > 0:
            self.mlp = nn.Sequential(
                nn.Linear(in_feats, hidden), nn.GELU(),
                nn.Linear(hidden, K + 1)   # K logits + 1 strength
            )
        else:
            self.mlp = nn.Linear(in_feats, K + 1)

        # depthwise spatial gate -> per-pixel λ multiplier in [0,1]
        self.spatial_gate = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                      groups=channels, bias=True)
        nn.init.constant_(self.spatial_gate.bias, gate_bias)  # start ~low regularization after sigmoid

        # positivity for strength
        self.softplus = nn.Softplus()

        # (Optional) slightly bias initial mixture toward Anisotropic, and set initial strength
        with torch.no_grad():
            last = self.mlp[-1] if isinstance(self.mlp, nn.Sequential) else self.mlp
            if last.bias is not None and last.bias.numel() == (K + 1):
                # logits bias: [TV, Lap, Aniso]
                b = last.bias.clone()
                b[:K] = 0.0
                b[K-1] = init_aniso_bias
                # strength bias so Softplus ≈ init_strength
                b[K] = torch.log(torch.tensor(init_strength).exp() - 1.0)
                last.bias.copy_(b)

    def _stats(self, x):
        # x: [B,C,H,W] -> stats: [B,C,6]
        gap = self.pool_gap(x).squeeze(-1).squeeze(-1)     # [B,C]
        gmp = self.pool_gmp(x).squeeze(-1).squeeze(-1)     # [B,C]

        gx, gy = _grad_x(x), _grad_y(x)
        gmean = (gx.abs().mean(dim=(-1,-2)) + gy.abs().mean(dim=(-1,-2))) * 0.5

        k = torch.tensor([[0., 1., 0.],[1., -4., 1.],[0., 1., 0.]],
                         device=x.device, dtype=x.dtype).view(1,1,3,3)
        lap = F.conv2d(x, k.repeat(x.size(1),1,1,1), padding=1, groups=x.size(1))
        lmean = lap.abs().mean(dim=(-1,-2))

        var  = x.var(dim=(-1,-2), unbiased=False)
        A = (gx * gx).mean(dim=(-1,-2)); B = (gy * gy).mean(dim=(-1,-2))
        aniso = (A - B).abs() / (A + B + 1e-6)

        return torch.stack([gap, gmp, gmean, lmean, var, aniso], dim=-1)  # [B,C,6]

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) per-channel stats -> mixture & strength
        s = self._stats(x)                                  # [B,C,6]
        head = self.mlp(s)                                  # [B,C,K+1]
        logits = head[..., :self.K] / max(self.temperature, 1e-6)  # [B,C,K]
        raw_strength = head[..., -1]                        # [B,C]

        if self.use_gumbel:
            mix = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
        else:
            mix = torch.softmax(logits, dim=-1)             # [B,C,K]

        lam_c = self.softplus(raw_strength).unsqueeze(-1)   # [B,C,1]
        lam = (self.base_lambda * lam_c).view(B, C, 1, 1)   # [B,C,1,1]

        # 2) spatial gate -> per-pixel λ
        g_sp = torch.sigmoid(self.spatial_gate(x))          # [B,C,H,W] in [0,1]
        lam_full = lam * g_sp                               # [B,C,H,W]
        self._g_sp = g_sp.detach()                          # (optional) inspect later

        # 3) apply T steps of mixed regularizers
        out = x
        for _ in range(self.steps):
            y_tv  = tv_step(out, lam_full)
            y_lap = laplacian_step(out, lam_full)
            y_ad  = anisotropic_diffusion_step(out, lam_full, kappa=self.kappa)

            # stack & mix: [B,C,K,H,W]  x  [B,C,K,1,1]
            Y = torch.stack([y_tv, y_lap, y_ad], dim=2)
            w = mix.unsqueeze(-1).unsqueeze(-1)
            out = (Y * w).sum(dim=2)

        # keep return signature compatible with your logging
        return out, mix.detach(), lam.detach()
