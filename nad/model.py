"""
model.py -- 제안 모델 (CLAUDE.md 3.1, 3.2)

스테레오 L,R ─┬─ (A) 공간 특징 7채널 -> ResNet -> Conformer(4층) ────────┐
              └─ (B) CLAP(고정, 캐시) -> 256 ─── CMC(Q=공간, K/V=의미) ┴-> 음향 토큰 200×256 ─┐
영상 ─ (C) OWL-ViT 열 특징 200×24×77 + 열 방위각 -> 열 토큰 200×24×256 ─────────────────────┤
          └ (D) 회전 추정 (t-1, t 열 특징) -> Δψ, 회전 특징 32 ──────────────────────────────┤
(E-1) 국소 교차 어텐션: 음향 t -> 영상 t-2..t+2 열 토큰 (120개), 잔차 ─────────────────────────┘
(E-2) [AV 256 ; 회전 32] -> 256 + 위치 인코딩 -> Conformer(4층)
(F) 트랙3 × [활성, sinθ, 앞/뒤, log 거리, onscreen] × 클래스13 = 195 (활성 함수 전 값)
    W1(loss_version=w1)에서는 sinθ 자리가 접은 방위각 α/20° (선형, 가두지 않음)이다. 모델 구조는 같고 해석만 다르다.

1단계(audio): 영상 입력이 없으면 (C),(D),(E-1)을 건너뛰고 회전 특징은 0 이다. 2단계는 1단계 가중치를 그대로 이어받는다.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import AcousticFeatures, N_CH
from .config import NB_CLASSES, TRACKS

N_ELEM = 5        # 활성, sinθ, 앞/뒤, log 거리, onscreen
E_ACT, E_SIN, E_FB, E_DIST, E_ON = range(5)


class PosEnc(nn.Module):
    def __init__(self, d, max_len=2000):
        super().__init__()
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1]].to(x.dtype)


class FFN(nn.Module):
    def __init__(self, d, drop, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, mult * d), nn.SiLU(), nn.Dropout(drop), nn.Linear(mult * d, d), nn.Dropout(drop))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, d, k, drop):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):                         # (B, T, d)
        y = self.ln(x).transpose(1, 2)
        y = F.glu(self.pw1(y), dim=1)
        y = F.silu(self.bn(self.dw(y)))
        return self.drop(self.pw2(y).transpose(1, 2))


class ConformerBlock(nn.Module):
    """FFN/2 -> (자기 또는 교차) 어텐션 -> 합성곱 -> FFN/2 -> LN. cross=True 면 K/V 가 mem 이다(CMC)."""
    def __init__(self, d, heads, k, drop, cross=False):
        super().__init__()
        self.cross = cross
        self.ff1, self.ff2 = FFN(d, drop), FFN(d, drop)
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d) if cross else None
        self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.conv = ConvModule(d, k, drop)
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x, mem=None):
        x = x + 0.5 * self.ff1(x)
        q = self.ln_q(x)
        kv = self.ln_kv(mem) if self.cross else q
        x = x + self.drop(self.att(q, kv, kv, need_weights=False)[0])
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, pool):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout)) if cin != cout else nn.Identity()
        self.pool = nn.AvgPool2d(pool) if pool != (1, 1) else nn.Identity()

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return self.pool(F.relu(y + self.sc(x)))


class SpatialEncoder(nn.Module):
    """(A) 7채널 (B, 7, 1601, 64) -> ResNet (시간 /8, 주파수 /16 후 평균) -> (B, 200, d) -> Conformer"""
    def __init__(self, d, heads, layers, k, drop, T_out=200):
        super().__init__()
        self.T_out = T_out
        self.stem = nn.Sequential(nn.Conv2d(N_CH, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(64, 64, (2, 2)), ResBlock(64, 128, (2, 2)), ResBlock(128, 256, (2, 2)), ResBlock(256, 256, (1, 2)))
        self.proj = nn.Linear(256, d)
        self.pe = PosEnc(d)
        self.drop = nn.Dropout(drop)
        self.layers = nn.ModuleList([ConformerBlock(d, heads, k, drop) for _ in range(layers)])

    def forward(self, x):
        x = self.blocks(self.stem(x))                 # (B, 256, T', 4)
        x = x.mean(dim=3).transpose(1, 2)              # 주파수 축 평균 -> (B, T', 256)
        if x.shape[1] != self.T_out:                   # 1601 -> 200 (시간 풀링 8), 혹시 다르면 맞춘다
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.T_out).transpose(1, 2)
        x = self.drop(self.pe(self.proj(x)))
        for l in self.layers:
            x = l(x)
        return x


class VisualTokens(nn.Module):
    """(C) 열 특징 (B, T, 24, 77) -> 표준화 -> 선형 + 열 방위각 (sinθ, cosθ) 선형 -> (B, T, 24, d)"""
    def __init__(self, d, vdim=NB_CLASSES + 64, ncol=24):
        super().__init__()
        self.register_buffer('vis_mean', torch.zeros(vdim))
        self.register_buffer('vis_std', torch.ones(vdim))
        self.register_buffer('col_az', torch.zeros(ncol))          # 라디안 (캐시 meta.json 에서 채운다)
        self.proj = nn.Linear(vdim, d)
        self.az = nn.Linear(2, d)
        self.ln = nn.LayerNorm(d)

    def standardize(self, v):
        return (v.float() - self.vis_mean) / self.vis_std

    def tokens(self, z):
        pos = torch.stack([torch.sin(self.col_az), torch.cos(self.col_az)], -1)       # (24, 2)
        return self.ln(self.proj(z) + self.az(pos))

    def forward(self, v):
        z = self.standardize(v)
        return self.tokens(z), z


class RotationHead(nn.Module):
    """(D) 프레임 t-1 과 t 의 열 특징 -> MLP -> Δψ (도/프레임), 회전 특징 32. 첫 프레임은 (t, t+1)."""
    def __init__(self, rot_dim=32, vdim=NB_CLASSES + 64, ncol=24, c=32, hidden=256, drop=0.1, with_pitch=False):
        super().__init__()
        self.inp = nn.Linear(vdim, c)
        self.mlp = nn.Sequential(nn.Linear(2 * ncol * c, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, hidden), nn.GELU())
        self.out_dpsi = nn.Linear(hidden, 2 if with_pitch else 1)
        self.out_feat = nn.Linear(hidden, rot_dim)

    def forward(self, z):                              # z: 표준화한 열 특징 (B, T, 24, 77)
        x = self.inp(z).flatten(2)                     # (B, T, 24*c)
        a = torch.cat([x[:, :1], x[:, :-1]], 1)        # t=0: x0, t>=1: x_{t-1}
        b = torch.cat([x[:, 1:2], x[:, 1:]], 1)        # t=0: x1, t>=1: x_t
        h = self.mlp(torch.cat([a, b], -1))
        return self.out_dpsi(h)[..., 0], self.out_feat(h)


class LocalCrossAttention(nn.Module):
    """(E-1) Q=음향 프레임 t, K/V=영상 프레임 t-k..t+k 의 열 토큰((2k+1)×24). 시간 오프셋 임베딩을 더한다. 결과를 잔차로 더한다."""
    def __init__(self, d, heads, k, drop):
        super().__init__()
        self.k = k
        self.off = nn.Parameter(torch.zeros(2 * k + 1, 1, d))
        self.ln_q, self.ln_kv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.drop = nn.Dropout(drop)

    def forward(self, a, v):                           # a: (B, T, d), v: (B, T, C, d)
        B, T, C, d = v.shape
        k = self.k
        vp = F.pad(v, (0, 0, 0, 0, k, k))                                      # 시간 양쪽 k 프레임 0 채움
        win = torch.stack([vp[:, i:i + T] for i in range(2 * k + 1)], dim=2)   # (B, T, 2k+1, C, d)
        win = (win + self.off.to(win.dtype)).reshape(B * T, (2 * k + 1) * C, d)
        t = torch.arange(T, device=v.device)[:, None] + torch.arange(-k, k + 1, device=v.device)[None]   # (T, 2k+1)
        pad = ((t < 0) | (t >= T))[:, :, None].expand(T, 2 * k + 1, C).reshape(1, T, -1).expand(B, T, -1).reshape(B * T, -1)
        q = self.ln_q(a).reshape(B * T, 1, d)
        kv = self.ln_kv(win)
        out = self.att(q, kv, kv, key_padding_mask=pad, need_weights=False)[0].reshape(B, T, d)
        return a + self.drop(out)


class NADModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h, k, dr = cfg['d_model'], cfg['n_heads'], cfg['conv_kernel'], cfg['dropout']
        self.rot_dim = cfg['rot_dim']
        self.feat = AcousticFeatures(cfg['sr'], cfg['n_fft'], cfg['hop'], cfg['n_mels'], cfg['fmin'], cfg['msc_lambda'], cfg['div_lag_s'])
        self.spatial = SpatialEncoder(d, h, cfg['spatial_layers'], k, dr)
        self.sem_proj = nn.Sequential(nn.Linear(512, d), nn.LayerNorm(d))
        self.sem_pe = PosEnc(d)
        self.cmc = ConformerBlock(d, h, k, dr, cross=True)
        # 영상 쪽 (2단계)
        self.vis = VisualTokens(d)
        self.rot = RotationHead(self.rot_dim, drop=dr)
        self.lca = LocalCrossAttention(d, h, cfg['av_window'], dr)
        # 시간 Conformer와 출력
        self.fuse = nn.Linear(d + self.rot_dim, d)
        self.t_pe = PosEnc(d)
        self.t_drop = nn.Dropout(dr)
        self.temporal = nn.ModuleList([ConformerBlock(d, h, k, dr) for _ in range(cfg['temporal_layers'])])
        self.head = nn.Linear(d, TRACKS * N_ELEM * NB_CLASSES)
        # W1 영상 정규화 (2단계 학습 때만): 열 드롭아웃, (D) 입력 노이즈
        w1v = cfg.get('loss_version', 'v0') == 'w1' and cfg.get('stage') == 2
        self.col_drop = float(cfg.get('col_drop', 0.0)) if w1v else 0.0
        self.d_noise = float(cfg.get('d_noise', 0.0)) if w1v else 0.0

    def acoustic(self, stereo, clap):
        x = self.feat(stereo)                                                  # (B, 7, 1601, 64) 정규화됨
        sp = self.spatial(x)                                                   # (B, 200, d)
        se = self.sem_pe(self.sem_proj(clap.float()))                          # (B, 200, d)
        return self.cmc(sp, se)

    def forward(self, stereo, clap, video=None, vkeep=None):
        """stereo (B,2,N), clap (B,200,512), video (B,200,24,77) 또는 None.
        vkeep: (B,) 0/1. 0 인 샘플은 표준화한 영상 특징 전체를 0 으로 둔다(W1 영상 드롭아웃, D 입력 포함).
        반환: dict(out=(B,T,3,5,13) 활성 함수 전 값, dpsi=(B,T) 또는 None)"""
        a = self.acoustic(stereo, clap)
        B, T, _ = a.shape
        dpsi = None
        if video is not None:
            z = self.vis.standardize(video)                                    # (B, T, 24, 77)
            if vkeep is not None:
                z = z * vkeep.to(z.dtype)[:, None, None, None]
            if self.training and self.col_drop > 0:
                z = z * (torch.rand(B, z.shape[1], z.shape[2], 1, device=z.device) >= self.col_drop).to(z.dtype)
            vt = self.vis.tokens(z)
            zr = z
            if self.training and self.d_noise > 0:                             # 표준화 특징의 표준편차는 1 이므로 노이즈 σ = d_noise
                n = torch.randn_like(z) * self.d_noise
                zr = z + (n if vkeep is None else n * vkeep.to(z.dtype)[:, None, None, None])
            dpsi, rf = self.rot(zr)
            a = self.lca(a, vt)
        else:
            rf = a.new_zeros(B, T, self.rot_dim)
        x = self.t_drop(self.t_pe(self.fuse(torch.cat([a, rf], -1))))
        for l in self.temporal:
            x = l(x)
        out = self.head(x).view(B, T, TRACKS, N_ELEM, NB_CLASSES)
        return dict(out=out, dpsi=dpsi)

    def visual_modules(self):
        return [self.vis, self.rot, self.lca]

    def set_visual_meta(self, meta):
        self.vis.vis_mean.copy_(torch.tensor(meta['vis_mean']))
        self.vis.vis_std.copy_(torch.tensor(meta['vis_std']))
        self.vis.col_az.copy_(torch.deg2rad(torch.tensor(meta['col_az_deg'], dtype=torch.float32)))
