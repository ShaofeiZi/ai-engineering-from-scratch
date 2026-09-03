import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoPatch3D(nn.Module):
    def __init__(self, in_channels=4, dim=64, patch_t=2, patch_h=2, patch_w=2):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, dim,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w),
        )

    def forward(self, x):
        x = self.proj(x)
        n, c, t, h, w = x.shape
        tokens = x.reshape(n, c, t * h * w).transpose(1, 2)
        return tokens, (t, h, w)


class DividedAttentionBlock(nn.Module):
    def __init__(self, dim=64, heads=2):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.space_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, grid):
        T, H, W = grid
        n, seq, d = x.shape

        xt = x.view(n, T, H * W, d).permute(0, 2, 1, 3).reshape(n * H * W, T, d)
        a, _ = self.time_attn(self.ln1(xt), self.ln1(xt), self.ln1(xt), need_weights=False)
        xt = (xt + a).reshape(n, H * W, T, d).permute(0, 2, 1, 3).reshape(n, seq, d)

        xs = xt.view(n, T, H * W, d).reshape(n * T, H * W, d)
        a, _ = self.space_attn(self.ln2(xs), self.ln2(xs), self.ln2(xs), need_weights=False)
        xs = (xs + a).reshape(n, T, H * W, d).reshape(n, seq, d)

        xs = xs + self.mlp(self.ln3(xs))
        return xs


class TinyVideoDiT(nn.Module):
    def __init__(self, in_channels=4, dim=64, depth=2, heads=2):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.patch = VideoPatch3D(in_channels=in_channels, dim=dim, patch_t=2, patch_h=2, patch_w=2)
        self.blocks = nn.ModuleList([DividedAttentionBlock(dim, heads) for _ in range(depth)])
        self.out = nn.Linear(dim, in_channels * 2 * 2 * 2)

    def forward(self, x):
        tokens, grid = self.patch(x)
        for blk in self.blocks:
            tokens = blk(tokens, grid)
        return self.out(tokens), grid


def count_tokens(T, H, W, p_t=2, p_h=8, p_w=8):
    return (T // p_t) * (H // p_h) * (W // p_w)


def main():
    print("[5 秒 360p 视频的 token 数（150 帧，480x360）]")
    tokens = count_tokens(150, 480, 360, p_t=2, p_h=8, p_w=8)
    T_tok = 150 // 2
    S_tok = (480 // 8) * (360 // 8)
    print(f"  每个片段的 token 数：{tokens:,}")
    print(f"  注意力对数（联合）：{tokens ** 2:,}")
    # 分解式时间注意力：在每个空间位置计算 T^2 次注意力。
    # 分解式空间注意力：在每个时间步计算 (H*W)^2 次注意力。
    divided_time = S_tok * T_tok ** 2
    divided_space = T_tok * S_tok ** 2
    print(f"  分解式时间注意力总量：{divided_time:,}")
    print(f"  分解式空间注意力总量：{divided_space:,}")
    print(f"  分解式注意力总量：{divided_time + divided_space:,}")

    torch.manual_seed(0)
    vid = torch.randn(1, 4, 8, 16, 16)
    model = TinyVideoDiT(in_channels=4, dim=64, depth=2, heads=2)
    out, grid = model(vid)
    print(f"\n[模型形状]")
    print(f"  输入       {tuple(vid.shape)}")
    print(f"  token 网格 {grid}")
    print(f"  输出       {tuple(out.shape)}")
    print(f"  参数量     {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
