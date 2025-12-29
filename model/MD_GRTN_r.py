# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from lib.utils import scaled_Laplacian


# ===== Time embedding（DDPM 标准）=====
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,)
        """
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


############################################
# 1. Diffusion Denoiser (DDPM + U-Net)
############################################
class DiffusionDenoiser(nn.Module):
    """
    DDPM-based denoiser

    输入 :
        x0 : (B, N, F, T)
    输出 :
        x_denoised : (B, N, D, T)
        diffusion_loss : scalar
    """
    def __init__(
        self,
        F_in,
        D,
        diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2
    ):
        super().__init__()

        self.D = D
        self.diffusion_steps = diffusion_steps

        # -------- β schedule --------
        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        # -------- Time embedding --------
        self.time_embed = SinusoidalTimeEmbedding(D)
        self.time_mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

        # -------- U-Net backbone (1D, time axis) --------
        self.enc1 = nn.Conv1d(F_in, D, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(D, D, kernel_size=3, padding=1)

        self.dec1 = nn.Conv1d(D, D, kernel_size=3, padding=1)
        self.dec2 = nn.Conv1d(D, D, kernel_size=3, padding=1)

    def forward(self, x0):
        """
        x0 : (B,N,F,T)
        """
        B, N, F, T = x0.shape
        x0 = x0.reshape(B * N, F, T)

        # -------- sample diffusion step t --------
        t = torch.randint(
            0, self.diffusion_steps,
            (B * N,), device=x0.device
        )

        # -------- forward diffusion (公式 2) --------
        eps = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)

        x_t = (
            torch.sqrt(alpha_bar_t) * x0 +
            torch.sqrt(1.0 - alpha_bar_t) * eps
        )

        # -------- time conditioning --------
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb).unsqueeze(-1)  # (B*N,D,1)

        # -------- U-Net εθ(x_t, t) --------
        h1 = F.relu(self.enc1(x_t))
        h2 = F.relu(self.enc2(h1 + t_emb))

        h3 = F.relu(self.dec1(h2))
        eps_hat = self.dec2(h3 + h1)   # 预测噪声 ε̂

        # -------- DDPM loss --------
        diffusion_loss = F.mse_loss(eps_hat, eps)

        # -------- reverse denoising (估计 x0) --------
        x0_hat = (
            x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_hat
        ) / torch.sqrt(alpha_bar_t)

        x0_hat = x0_hat.reshape(B, N, self.D, T)

        return x0_hat, diffusion_loss



############################################
# 2. MDAF: Multi-period Diffusion Attention Fusion
############################################
class MDAF(nn.Module):
    """
    输入 :
        X_rec  : (B,N,F,T)
        X_hour : (B,N,F,T)
        X_day  : (B,N,F,T)
    输出 :
        X_mdaf : (B,N,D,T)
    """
    def __init__(self, F_in, D, nhead=1):
        super().__init__()

        # -------- Diffusion Denoisers (U-Net) --------
        self.rec = DiffusionDenoiser(F_in, D)
        self.hour = DiffusionDenoiser(F_in, D)
        self.day = DiffusionDenoiser(F_in, D)

        # -------- Temporal Self-Attention (公式 5–8) --------
        self.attn_rec = nn.MultiheadAttention(
            embed_dim=D, num_heads=nhead, batch_first=True
        )
        self.attn_hour = nn.MultiheadAttention(
            embed_dim=D, num_heads=nhead, batch_first=True
        )
        self.attn_day = nn.MultiheadAttention(
            embed_dim=D, num_heads=nhead, batch_first=True
        )

        # -------- Multi-head Fusion (公式 9) --------
        self.fusion = nn.Linear(3 * D, D)

    def forward(self, x_rec, x_hour, x_day):
        # -------- 1. Diffusion denoising --------
        xr, loss_r = self.rec(x_rec)    # (B,N,D,T)
        xh, loss_h = self.hour(x_hour)
        xd, loss_d = self.day(x_day)

        diffusion_loss = loss_r + loss_h + loss_d

        B, N, D, T = xr.shape
        # -------- 2. reshape for temporal attention --------
        xr = xr.permute(0, 1, 3, 2).reshape(B * N, T, D)
        xh = xh.permute(0, 1, 3, 2).reshape(B * N, T, D)
        xd = xd.permute(0, 1, 3, 2).reshape(B * N, T, D)

        # -------- 3. Temporal self-attention --------
        xr_attn, _ = self.attn_rec(xr, xr, xr)
        xh_attn, _ = self.attn_hour(xh, xh, xh)
        xd_attn, _ = self.attn_day(xd, xd, xd)

        # -------- 4. Concat + fusion --------
        x_cat = torch.cat([xr_attn, xh_attn, xd_attn], dim=-1)  # (B*N,T,3D)
        x_fused = self.fusion(x_cat)                            # (B*N,T,D)

        # -------- 5. reshape back --------
        x_fused = x_fused.reshape(B, N, T, D).permute(0, 1, 3, 2)

        return x_fused    # (B,N,D,T)



############################################
# 3. MGRC: Multi-Graph Recurrent Convolution
############################################
class MGRC(nn.Module):
    """
    输入 : (B,N,D,T)
    输出 : (B,N,D,T)
    """
    def __init__(self, num_nodes, D, adj_mx, DEVICE):
        super().__init__()
        self.N = num_nodes
        self.D = D
        self.DEVICE = DEVICE

        # 静态图
        self.Adist = torch.tensor(adj_mx, dtype=torch.float32, device=DEVICE)

        # 动态图参数（论文中的 E1, E2）
        self.E1 = nn.Parameter(torch.randn(num_nodes, D))
        self.E2 = nn.Parameter(torch.randn(num_nodes, D))

        # 递归建模时间
        self.gru = nn.GRU(D, D, batch_first=True)

    def forward(self, x):
        # x: (B,N,D,T)
        B, N, D, T = x.shape

        # 动态邻接矩阵
        Adyna = torch.softmax(
            torch.relu(self.E1 @ self.E2.T),
            dim=-1
        )                          # (N,N)

        A = self.Adist + Adyna     # (N,N)

        x = x.permute(0, 3, 1, 2)  # (B,T,N,D)

        gcn_out = []
        for t in range(T):
            xt = x[:, t]           # (B,N,D)
            xt = torch.matmul(A, xt)  # (B,N,D)
            gcn_out.append(xt)

        x = torch.stack(gcn_out, dim=1)   # (B,T,N,D)

        # GRU 建模时间递归
        x = x.reshape(B*T, N, D)
        x, _ = self.gru(x)
        x = x.reshape(B, T, N, D)

        return x.permute(0, 2, 3, 1)  # (B,N,D,T)


############################################
# 4. STFormer: Spatial-Temporal Transformer
############################################
class STFormer(nn.Module):
    """
    输入 : (B,N,D,T)
    输出 : (B,N,D,T)
    """
    def __init__(self, D, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        # x: (B,N,D,T)
        B, N, D, T = x.shape
        x = x.permute(0, 1, 3, 2)      # (B,N,T,D)
        x = x.reshape(B*N, T, D)
        x = self.encoder(x)
        x = x.reshape(B, N, T, D)
        return x.permute(0, 1, 3, 2)  # (B,N,D,T)


############################################
# 5. MD-GRTN 主模型
############################################
class MD_GRTN(nn.Module):
    """
    输入 : (B,N,F,T)
    输出 : (B,N,T_out)
    """
    def __init__(self, DEVICE, num_nodes, F_in, D, T_out, adj_mx):
        super().__init__()
        self.mdaf = MDAF(F_in, D)
        self.mgrc = MGRC(num_nodes, D, adj_mx, DEVICE)
        self.stformer = STFormer(D, nhead=4, num_layers=2)

        self.proj = nn.Conv2d(
            in_channels=D,
            out_channels=T_out,
            kernel_size=(1, 1)
        )

        self.to(DEVICE)

    def forward(self, x):
        # x: (B,N,F,T)
        x = self.mdaf(x, x, x)      # (B,N,D,T)
        x = self.mgrc(x)            # (B,N,D,T)
        x = self.stformer(x)        # (B,N,D,T)
        x = self.proj(x)            # (B,N,T_out,T)
        return x[..., -1]           # (B,N,T_out)


############################################
# 6. make_model（保持与 ASTGCN 一致）
############################################
def make_model(
    DEVICE,
    num_nodes,
    F_in,
    D,
    T_out,
    adj_mx
):
    model = MD_GRTN(
        DEVICE=DEVICE,
        num_nodes=num_nodes,
        F_in=F_in,
        D=D,
        T_out=T_out,
        adj_mx=adj_mx
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
