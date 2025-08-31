import torch
from models.network_swinir import SwinIR

def round_up(n, m):  # smallest multiple of m >= n
    return ((n + m - 1) // m) * m

def run_case(upsampler, upscale, H=64, W=64, use_mixer=True, device=None, window_size=8):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # round img_size to multiple of window_size to satisfy mask precompute
    Hm, Wm = round_up(H, window_size), round_up(W, window_size)

    net = SwinIR(
        upscale=upscale,
        img_size=(Hm, Wm),            # MUST be multiple of window_size
        window_size=window_size,      # choose 8 if you keep H=W=64
        upsampler=upsampler,
        embed_dim=96,
        depths=[2,2,2,2],
        num_heads=[4,4,4,4],
        # mixer knobs
        use_ch_reg_mixer=use_mixer,
        ch_reg_steps=2,
        ch_reg_lambda=0.03,
        ch_reg_kappa=0.03,
        ch_reg_hidden=32,
        ch_reg_temp=1.0,
    ).to(device)

    net.train()
    x = torch.randn(2, 3, H, W, device=device)  # original size (no need to be multiple)
    y = net(x)
    if hasattr(net, "_mix_w"):
        mw = net._mix_w.mean(dim=(0, 1))  # [K]
        print("mix_w avg (TV, Lap, Aniso):", [float(v) for v in mw])

    if hasattr(net, "ch_reg_mixer") and hasattr(net.ch_reg_mixer, "_g_sp"):
        print("spatial gate mean/std:",
            float(net.ch_reg_mixer._g_sp.mean()),
            float(net.ch_reg_mixer._g_sp.std()))

    # mean channel lambda (base_lambda * Softplus(strength)), BEFORE spatial gating
    if hasattr(net, "_lam_c"):
        lc = float(net._lam_c.mean())
        print("mean lambda (channel):", lc)

        # Optional: effective lambda after spatial gate (what actually modulates the steps)
        if hasattr(net, "ch_reg_mixer") and hasattr(net.ch_reg_mixer, "_g_sp"):
            eff = float((net._lam_c * net.ch_reg_mixer._g_sp).mean())
            print("mean effective lambda (channel × spatial gate):", eff)

    print("mix_w per-regularizer (TV, Lap, Aniso):", [float(v) for v in mw])
    print("mean lambda:", float(lc))
    print(f"[{upsampler or 'denoise/jpeg'}] out shape: {tuple(y.shape)}")

    # grad check
    target = torch.randn_like(y)
    loss = (y - target).abs().mean()
    loss.backward()
    print("backward ok; first-param |grad| mean:",
          float(next(net.parameters()).grad.abs().mean()))

if __name__ == "__main__":
    run_case("pixelshuffle", 4, window_size=8)        # classic SR
    run_case("pixelshuffledirect", 4, window_size=8)  # lightweight SR
    run_case("nearest+conv", 4, window_size=8)        # real-world SR (must be x4)
    run_case("", 1, window_size=8)                    # denoise/JPEG (upscale=1)

