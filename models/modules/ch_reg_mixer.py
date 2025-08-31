import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- classical steps (feature-space) ----------
def _grad_x(u): return u[..., :, 1:] - u[..., :, :-1]
def _grad_y(u): return u[..., 1:, :] - u[..., :-1, :]

def _div(px, py):
    # adjoint of forward differences; simple Neumann boundaries
    divx = F.pad(px[..., :, :-1], (1,0,0,0)) - F.pad(px, (0,1,0,0))
    divy = F.pad(py[..., :-1, :], (0,0,1,0)) - F.pad(py, (0,0,0,1))
    H = min(divx.size(-2), divy.size(-2))
    W = min(divx.size(-1), divy.size(-1))
    return divx[..., :H, :W] + divy[..., :H, :W]

def tv_step(x, lam, step=0.25, eps=1e-6):
    # lam: [B,C,1,1] or [1,C,1,1]
    gx, gy = _grad_x(x), _grad_y(x)
    nx = gx / (gx.abs() + eps)
    ny = gy / (gy.abs() + eps)
    px = F.pad(nx, (0,1,0,0))
    py = F.pad(ny, (0,0,0,1))
    d = _div(px, py)
    return x - step * lam * d

def laplacian_step(x, lam, step=0.25):
    # 2D Laplacian via depthwise conv
    B, C, H, W = x.shape
    k = torch.tensor([[0., 1., 0.],
                      [1., -4., 1.],
                      [0., 1., 0.]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    weight = k.repeat(C, 1, 1, 1)  # depthwise
    lap = F.conv2d(x, weight, padding=1, groups=C)
    return x - step * lam * lap

def anisotropic_diffusion_step(x, lam, step=0.25, kappa=0.03, eps=1e-6):
    # Perona–Malik-ish: g(|∇x|)=1/(1+(|∇x|/kappa)^2)
    gx, gy = _grad_x(x), _grad_y(x)
    mag = torch.sqrt(gx*gx + gy*gy + eps)
    g = 1.0 / (1.0 + (mag / kappa)**2)
    px = F.pad(g*gx/(mag+eps), (0,1,0,0))
    py = F.pad(g*gy/(mag+eps), (0,0,0,1))
    d = _div(px, py)
    return x - step * lam * d

# --------- mixer module ----------
class ChannelRegMixer(nn.Module):
    """
    For each channel, predict a mixture over K regularizers and apply T steps.
    Regularizers (K=3): [TV, Laplacian (Tikhonov), Anisotropic diffusion]
    """
    def __init__(self, channels, steps=2, base_lambda=0.03, kappa=0.03, hidden=0, temperature=1.0):
        super().__init__()
        self.steps = steps
        self.base_lambda = base_lambda
        self.kappa = kappa
        self.temperature = temperature

        # Gating head: per-channel mixture over K regs
        K = 3
        self.K = K

        # stats extractor: per-channel global descriptors
        # [B,C,H,W] -> [B,C,4] using GAP/MAXP + mean(|grad|) + mean(|lap|)
        self.pool_gap = nn.AdaptiveAvgPool2d(1)
        self.pool_gmp = nn.AdaptiveMaxPool2d(1)

        # small head to produce per-channel logits for K regularizers and a per-channel strength
        in_feats = 4
        if hidden > 0:
            self.mlp = nn.Sequential(
                nn.Linear(in_feats, hidden), nn.GELU(),
                nn.Linear(hidden, K + 1)   # K logits + 1 strength
            )
        else:
            self.mlp = nn.Linear(in_feats, K + 1)

        # softness/positivity
        self.softplus = nn.Softplus()

    def _stats(self, x):
        # x: [B,C,H,W] -> stats: [B,C,4]
        gap = self.pool_gap(x).squeeze(-1).squeeze(-1)     # [B,C]
        gmp = self.pool_gmp(x).squeeze(-1).squeeze(-1)     # [B,C]
        # gradient magnitude mean
        gx, gy = _grad_x(x), _grad_y(x)
        gmean = (gx.abs().mean(dim=(-1,-2)) + gy.abs().mean(dim=(-1,-2))) * 0.5  # [B,C]
        # laplacian magnitude mean
        k = torch.tensor([[0., 1., 0.],[1., -4., 1.],[0., 1., 0.]],
                         device=x.device, dtype=x.dtype).view(1,1,3,3)
        lap = F.conv2d(x, k.repeat(x.size(1),1,1,1), padding=1, groups=x.size(1))
        lmean = lap.abs().mean(dim=(-1,-2))               # [B,C]
        return torch.stack([gap, gmp, gmean, lmean], dim=-1)  # [B,C,4]

    def forward(self, x):
        B, C, H, W = x.shape

        # per-channel stats -> gating + strength
        s = self._stats(x)                                  # [B,C,4]
        head = self.mlp(s)                                  # [B,C,K+1]
        logits = head[..., :self.K] / self.temperature      # [B,C,K]
        raw_strength = head[..., -1]                        # [B,C]

        mix = torch.softmax(logits, dim=-1)                 # convex weights per channel over K regs
        lam_c = self.softplus(raw_strength).unsqueeze(-1)   # [B,C,1] non-negative
        lam = (self.base_lambda * lam_c).view(B, C, 1, 1)   # [B,C,1,1]

        out = x
        for _ in range(self.steps):
            # compute each candidate step
            y_tv  = tv_step(out, lam)
            y_lap = laplacian_step(out, lam)
            y_ad  = anisotropic_diffusion_step(out, lam, kappa=self.kappa)

            # stack candidates: [B,C,K,H,W]
            Y = torch.stack([y_tv, y_lap, y_ad], dim=2)

            # mix per-channel with mix weights [B,C,K] -> [B,C,K,1,1]
            w = mix.unsqueeze(-1).unsqueeze(-1)
            out = (Y * w).sum(dim=2)

        return out, mix.detach(), lam.detach()