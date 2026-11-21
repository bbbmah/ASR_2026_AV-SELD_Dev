"""
features.py -- (A) 공간 경로의 음향 특징을 GPU 에서 계산한다.

채널 7개: L, R log-mel, Mid, Side log-mel, IV, MSC, ΔIV  ->  (B, 7, T=1601, 64)
계산식은 NTU 코드(for_dataset/baseline/ntu_challenge/utils.py)의 것과 같다:
 - log-mel: librosa.feature.melspectrogram(S=|X|^2, fmin=50) + librosa.power_to_db (ref=1, amin=1e-10, top_db=80)
 - IV (compute_msiv): Re(M·conj(S)) / (eps + |M|^2 + |S|^2) 를 mel 필터로 모음
 - MSC (compute_binaural_msc_recursive): λ=0.8 재귀 평균으로 S11, S22, S12 -> |S12|^2/(S11·S22) 를 mel 필터로 모음
   (재귀 평균은 길이 64 의 인과 합성곱으로 계산한다. 잘린 꼬리 λ^64 ≈ 6e-7)
 - 새로 추가: ΔIV[t] = IV[t] - IV[t - 1초] (처음 1초는 첫 프레임 값으로 채운다)
STFT 는 n_fft=1024, hop=300(12.5ms), 한 창, 가운데 맞춤(reflect). NTU 의 IV 만 librosa 기본 패딩(constant)이라 처음/끝 몇 프레임이 조금 다르다.
check_against_ntu() 로 한 클립에서 NTU numpy 결과와 비교한다.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CH = 7
CH_NAMES = ['L', 'R', 'Mid', 'Side', 'IV', 'MSC', 'dIV']
SHARED = 4          # 앞 4 채널(L,R,Mid,Side)은 정규화 통계를 공유한다 (NTU 와 같음)


class AcousticFeatures(nn.Module):
    def __init__(self, sr=24000, n_fft=1024, hop=300, n_mels=64, fmin=50, msc_lambda=0.8, div_lag_s=1.0, top_db=80.0, msc_taps=64):
        super().__init__()
        import librosa
        self.n_fft, self.hop, self.top_db = n_fft, hop, top_db
        self.div_lag = int(round(div_lag_s * sr / hop))
        self.register_buffer('mel', torch.tensor(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin), dtype=torch.float32))
        self.register_buffer('win', torch.hann_window(n_fft, periodic=True))
        k = (1 - msc_lambda) * msc_lambda ** np.arange(msc_taps)[::-1].copy()       # 인과 필터: y[t] = Σ (1-λ) λ^n x[t-n]
        self.register_buffer('msc_k', torch.tensor(k, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer('mean', torch.zeros(N_CH, n_mels))
        self.register_buffer('std', torch.ones(N_CH, n_mels))
        self.has_stats = False

    def _stft(self, x):                       # (B, N) -> (B, F, T) complex
        return torch.stft(x, self.n_fft, self.hop, self.n_fft, self.win, center=True, pad_mode='reflect', return_complex=True)

    def _db(self, S):                          # power_to_db(ref=1, amin=1e-10, top_db=80), 입력 하나(채널 하나)의 최댓값 기준
        x = 10.0 * torch.log10(torch.clamp(S, min=1e-10))
        return torch.maximum(x, x.amax(dim=(-2, -1), keepdim=True) - self.top_db)

    def _melmap(self, P):                       # (B, F, T) -> (B, T, 64)
        return torch.einsum('mf,bft->btm', self.mel, P)

    def _rec(self, x):                          # 재귀 평균, x: (B, F, T) 실수
        B, Fq, T = x.shape
        y = F.conv1d(F.pad(x.reshape(B * Fq, 1, T), (self.msc_k.shape[-1] - 1, 0)), self.msc_k)
        return y.view(B, Fq, T)

    @torch.no_grad()
    def raw(self, stereo):
        """stereo: (B, 2, N) float. 반환 (B, 7, T, 64) 정규화 전."""
        stereo = stereo.float()
        L, R = stereo[:, 0], stereo[:, 1]
        M, S = 0.5 * (L + R), 0.5 * (L - R)
        XL, XR, XM, XS = self._stft(L), self._stft(R), self._stft(M), self._stft(S)
        PL, PR, PM, PS = XL.abs() ** 2, XR.abs() ** 2, XM.abs() ** 2, XS.abs() ** 2
        chans = [self._db(self._melmap(P)) for P in (PL, PR, PM, PS)]
        iv = self._melmap(torch.real(XM * torch.conj(XS)) / (1e-8 + PM + PS))
        c12 = XL * torch.conj(XR)
        S11, S22 = self._rec(PL), self._rec(PR)
        S12r, S12i = self._rec(c12.real), self._rec(c12.imag)
        msc = self._melmap((S12r ** 2 + S12i ** 2) / (S11 * S22 + 1e-8))
        lag = self.div_lag
        prev = torch.cat([iv[:, :1].expand(-1, lag, -1), iv[:, :-lag]], dim=1) if iv.shape[1] > lag else iv[:, :1].expand_as(iv)
        div = iv - prev
        return torch.stack(chans + [iv, msc, div], dim=1)

    @torch.no_grad()
    def forward(self, stereo):
        x = self.raw(stereo)
        return (x - self.mean[None, :, None, :]) / self.std[None, :, None, :]

    @torch.no_grad()
    def fit_stats(self, batches):
        """batches: 스테레오 (B,2,N) 텐서들의 반복자. (채널, mel)별 평균/표준편차. 앞 4 채널은 통계 공유."""
        s = s2 = None
        n = 0
        for st in batches:
            x = self.raw(st.to(self.mel.device)).double()            # (B, 7, T, 64)
            s = x.sum(dim=(0, 2)) if s is None else s + x.sum(dim=(0, 2))
            s2 = (x ** 2).sum(dim=(0, 2)) if s2 is None else s2 + (x ** 2).sum(dim=(0, 2))
            n += x.shape[0] * x.shape[2]
        cnt = torch.full((N_CH, 1), float(n), dtype=torch.float64, device=s.device)
        s[:SHARED] = s[:SHARED].sum(0, keepdim=True); s2[:SHARED] = s2[:SHARED].sum(0, keepdim=True); cnt[:SHARED] = n * SHARED
        mean = s / cnt
        std = torch.sqrt(torch.clamp(s2 / cnt - mean ** 2, min=0))
        std[std == 0] = 1.0
        self.mean.copy_(mean.float()); self.std.copy_(std.float()); self.has_stats = True

    def stats_dict(self):
        return dict(mean=self.mean.cpu(), std=self.std.cpu())

    def load_stats(self, d):
        self.mean.copy_(d['mean']); self.std.copy_(d['std']); self.has_stats = True


def check_against_ntu(cfg, stereo_np, device='cpu', log=print):
    """한 클립(2, N)에서 GPU 특징과 NTU numpy 특징(extract_stereo_features + ΔIV)의 차이. 가장자리 4프레임은 뺀다."""
    from .config import import_ntu_utils
    nu = import_ntu_utils(cfg)
    hop = cfg['hop']; n_fft = 2 ** (2 * hop - 1).bit_length()
    ref = nu.extract_stereo_features(stereo_np.astype(np.float32), sr=cfg['sr'], n_fft=n_fft, hop_length=hop, win_length=n_fft,
                                     nb_mels=cfg['n_mels'], use_gamma=True, use_iv=True, use_ms=True)   # (L,R,M,S,IV,MSC)
    fe = AcousticFeatures(cfg['sr'], cfg['n_fft'], hop, cfg['n_mels'], cfg['fmin'], cfg['msc_lambda'], cfg['div_lag_s']).to(device)
    ours = fe.raw(torch.tensor(stereo_np[None], dtype=torch.float32, device=device))[0].cpu().numpy()
    out = {}
    for c in range(6):
        a, b = ours[c, 4:-4], ref[c, 4:-4]
        out[CH_NAMES[c]] = dict(max_abs=float(np.abs(a - b).max()), ref_range=float(np.abs(b).max()), T=(ours.shape[1], ref.shape[1]))
    log(out)
    return out
