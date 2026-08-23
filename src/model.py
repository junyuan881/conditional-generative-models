import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Convert timestep t into sinusoidal embedding.
    Input:
        t: shape (B,)
    Output:
        emb: shape (B, dim)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2

        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class ConditionEmbedding(nn.Module):
    """
    Convert multi-hot label vector into condition embedding.
    Input:
        cond: shape (B, num_classes)
    Output:
        emb: shape (B, cond_emb_dim)
    """

    def __init__(self, num_classes=24, cond_emb_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_classes, cond_emb_dim),
            nn.SiLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
        )

    def forward(self, cond):
        return self.net(cond.float())


class ResBlock(nn.Module):
    """
    Residual block with time-condition embedding.
    """

    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        emb_out = self.emb_proj(emb)
        h = h + emb_out[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """
    ResBlock + downsample.
    """

    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()

        self.res = ResBlock(in_channels, out_channels, emb_dim)
        self.down = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x, emb):
        h = self.res(x, emb)
        skip = h
        h = self.down(h)
        return h, skip


class UpBlock(nn.Module):
    """
    Upsample + concat skip + ResBlock.
    """

    def __init__(self, in_channels, skip_channels, out_channels, emb_dim):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.res = ResBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            emb_dim=emb_dim,
        )

    def forward(self, x, skip, emb):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)
        x = self.res(x, emb)

        return x


class ConditionalUNet(nn.Module):
    """
    Conditional UNet for DDPM noise prediction.

    Input:
        x:    noisy image x_t, shape (B, 3, 64, 64)
        t:    timestep, shape (B,)
        cond: multi-hot label, shape (B, 24)

    Output:
        predicted noise, shape (B, 3, 64, 64)
    """

    def __init__(
        self,
        img_channels=3,
        num_classes=24,
        base_channels=64,
        time_emb_dim=256,
        cond_emb_dim=256,
    ):
        super().__init__()

        self.img_channels = img_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = cond_emb_dim

        emb_dim = time_emb_dim

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.condition_embedding = ConditionEmbedding(
            num_classes=num_classes,
            cond_emb_dim=cond_emb_dim,
        )

        self.condition_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(
            img_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.down1 = DownBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            emb_dim=emb_dim,
        )

        self.down2 = DownBlock(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            emb_dim=emb_dim,
        )

        self.down3 = DownBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            emb_dim=emb_dim,
        )

        # Bottleneck
        self.mid1 = ResBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 4,
            emb_dim=emb_dim,
        )

        self.mid2 = ResBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 4,
            emb_dim=emb_dim,
        )

        # Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64
        self.up3 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 2,
            emb_dim=emb_dim,
        )

        self.up2 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels * 2,
            out_channels=base_channels,
            emb_dim=emb_dim,
        )

        self.up1 = UpBlock(
            in_channels=base_channels,
            skip_channels=base_channels,
            out_channels=base_channels,
            emb_dim=emb_dim,
        )

        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t, cond):
        """
        x: shape (B, 3, 64, 64)
        t: shape (B,)
        cond: shape (B, 24)
        """

        t_emb = self.time_embedding(t)
        c_emb = self.condition_embedding(cond)
        c_emb = self.condition_proj(c_emb)

        emb = t_emb + c_emb

        x = self.init_conv(x)

        x, skip1 = self.down1(x, emb)   # 64 -> 32
        x, skip2 = self.down2(x, emb)   # 32 -> 16
        x, skip3 = self.down3(x, emb)   # 16 -> 8

        x = self.mid1(x, emb)
        x = self.mid2(x, emb)

        x = self.up3(x, skip3, emb)     # 8 -> 16
        x = self.up2(x, skip2, emb)     # 16 -> 32
        x = self.up1(x, skip1, emb)     # 32 -> 64

        x = self.final(x)

        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConditionalUNet(
        img_channels=3,
        num_classes=24,
        base_channels=64,
        time_emb_dim=256,
        cond_emb_dim=256,
    ).to(device)

    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    cond = torch.zeros(batch_size, 24).to(device)
    cond[:, 0] = 1
    cond[:, 5] = 1

    with torch.no_grad():
        out = model(x, t, cond)

    print("Input shape:", x.shape)
    print("Timestep shape:", t.shape)
    print("Condition shape:", cond.shape)
    print("Output shape:", out.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")