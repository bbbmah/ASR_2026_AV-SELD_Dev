"""
loss.py -- 새 출력 헤드용 ADPIT 손실 (CLAUDE.md 3.3)

L = Σ_class min_ADPIT순열 Σ_트랙 [ BCE(a, â) + a · ( (sinθ - ŝ)² + |cosθ|·BCE(f, f̂) + |log d - ℓ̂| + BCE(o, ô) ) ]
    + λ · (Δψ - Δψ̂)²

 - 정답 트랙 복제 방식은 베이스라인 ADPIT(loss.py, Shimada)와 같다: 6개 더미 트랙 A0 | B0 B1 | C0 C1 C2, 순열 13개
   (A0A0A0, B 6개, C 6개). 쓰지 않는 그룹은 다른 그룹의 기본 배치를 더해 채운다(pad 트릭). 한 프레임·클래스에서
   A/B/C 중 한 그룹만 값이 있으므로, 더하면 "값이 있는 그룹의 배치"가 된다(활성값 포함).
 - 베이스라인은 ACCDOA MSE 하나로 순열을 고르지만, 여기서는 순열마다 위 손실 전체를 계산해 가장 작은 것을 고른다.
 - f = 1[cosθ > 0] (앞일 확률의 정답), 거리 정답은 log(미터).
 - 프리셋에 따라 앞/뒤(fb), onscreen, 회전 항을 끈다(1단계: 활성, sinθ, 거리만).
정답 텐서는 베이스라인 utils.process_labels_adpit 의 출력 (B, T, 6, 5[act, x=cosθ, y=sinθ, dist(m), onscreen], 13).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import E_ACT, E_SIN, E_FB, E_DIST, E_ON

_A = [[0, 0, 0]]
_B = [[1, 1, 2], [1, 2, 1], [1, 2, 2], [2, 1, 1], [2, 1, 2], [2, 2, 1]]
_C = [[3, 4, 5], [3, 5, 4], [4, 3, 5], [4, 5, 3], [5, 3, 4], [5, 4, 3]]
BASE = dict(A=[0, 0, 0], B=[1, 1, 2], C=[3, 4, 5])
PERMS = [(p, 'A') for p in _A] + [(p, 'B') for p in _B] + [(p, 'C') for p in _C]


def perm_targets(t):
    """t: (B, T, 6, 13) 한 요소 -> (13, B, T, 3, 13) 순열별 트랙 정답 (pad 트릭 포함)"""
    out = []
    for p, fam in PERMS:
        x = t[:, :, p]
        for other in ('A', 'B', 'C'):
            if other != fam:
                x = x + t[:, :, BASE[other]]
        out.append(x)
    return torch.stack(out)


class NADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fb, self.on, self.rot = cfg['loss_fb'], cfg['loss_onscreen'], cfg['loss_rot']
        self.lam = cfg['lambda_rot']

    def forward(self, pred, target, dpsi=None, vkeep=None, step=None):
        """(vkeep, step 은 W1 손실과 호출 형태를 맞추려고 받기만 한다)
        pred: dict(out=(B,T,3,5,13), dpsi=(B,T)|None), target: (B,T,6,5,13), dpsi: (B,T) 도/프레임 정답.
        반환: (전체 손실, 항목별 값 dict)"""
        out = pred['out'].float()
        act = perm_targets(target[:, :, :, 0])                     # (P, B, T, 3, C)
        cos = perm_targets(target[:, :, :, 1])
        sin = perm_targets(target[:, :, :, 2])
        dist = perm_targets(target[:, :, :, 3])
        ons = perm_targets(target[:, :, :, 4])
        logd = torch.log(dist.clamp(min=0.05))

        a_l = out[:, :, :, E_ACT][None]                            # (1, B, T, 3, C)
        s_p = torch.tanh(out[:, :, :, E_SIN])[None]
        f_l = out[:, :, :, E_FB][None]
        d_p = out[:, :, :, E_DIST][None]
        o_l = out[:, :, :, E_ON][None]

        l_act = F.binary_cross_entropy_with_logits(a_l.expand_as(act), act, reduction='none')
        l_sin = (sin - s_p) ** 2
        l_dist = (logd - d_p).abs()
        per = l_act + act * (l_sin + l_dist)
        l_fb = l_on = None
        if self.fb:
            l_fb = cos.abs() * F.binary_cross_entropy_with_logits(f_l.expand_as(cos), (cos > 0).float(), reduction='none')
            per = per + act * l_fb
        if self.on:
            l_on = F.binary_cross_entropy_with_logits(o_l.expand_as(ons), ons, reduction='none')
            per = per + act * l_on
        tot = per.sum(dim=3)                                       # 트랙 합 -> (P, B, T, C)
        best = tot.argmin(dim=0)                                   # 순열 선택 기준 = 위 항들의 합
        loss = torch.gather(tot, 0, best[None]).squeeze(0).sum(dim=-1).mean()    # 클래스 합, (배치, 프레임) 평균

        # 기록용 항목별 값 (고른 순열 기준, 활성 트랙 평균)
        with torch.no_grad():
            def pick(x):
                idx = best[None, :, :, None, :].expand(1, *x.shape[1:])
                return torch.gather(x, 0, idx).squeeze(0)
            a_sel = pick(act); n_act = a_sel.sum().clamp(min=1)
            parts = dict(act=float(pick(l_act).mean()), sin=float((pick(l_sin) * a_sel).sum() / n_act), dist=float((pick(l_dist) * a_sel).sum() / n_act))
            if l_fb is not None:
                parts['fb'] = float((pick(l_fb) * a_sel).sum() / n_act)
            if l_on is not None:
                parts['on'] = float((pick(l_on) * a_sel).sum() / n_act)
        if self.rot and pred.get('dpsi') is not None and dpsi is not None:
            l_rot = F.mse_loss(pred['dpsi'].float(), dpsi.float())
            loss = loss + self.lam * l_rot
            parts['rot'] = float(l_rot.detach())
        parts['total'] = float(loss.detach())
        return loss, parts


class W1Loss(nn.Module):
    """MODEL_DEV_2.md W1 손실.
    L = Σ_{t,c} min_π Σ_k [ w_a·ℓ_a + a·( w_s·ℓ_s + w_f·ℓ_f + w_d·ℓ_d + w_o·ℓ_o ) ] + w_ψ·ℓ_Δψ
      ① ℓ_a = BCE(a, â)                         전체 칸(배치, 프레임, 트랙, 클래스) 평균
      ② ℓ_s = Huber(δ=1)(α̂/20° 출력 - α/20°)     α = arcsin(sinθ) (접은 방위각). 출력 자리는 기존 sinθ 자리, 선형
      ③ ℓ_f = |cosθ|·BCE(f_ε, f̂), f_ε = 1-ε/2 (앞) | ε/2 (뒤)
      ④ ℓ_d = |ln d - ℓ̂| / ln2
      ⑤ ℓ_o = BCE(o_τ, ô)  (o_τ = 소프트 onscreen 라벨, 라벨 텐서의 onscreen 자리)
      ⑥ ℓ_Δψ = ((Δψ - Δψ̂)/1.8°)², 프레임 평균
    ②~⑤ 는 고른 순열 기준 a=1 칸의 수로 나눈다(0개면 0). ③⑤⑥ 은 영상 드롭 샘플(vkeep=0)을 빼고 계산하고 분모에서도 뺀다.
    순열은 위 가중 합(칸별, 트랙 합)으로 고른다(기존과 같은 방식)."""
    def __init__(self, cfg):
        super().__init__()
        self.fb, self.on, self.rot = cfg['loss_fb'], cfg['loss_onscreen'], cfg['loss_rot']
        self.c = {k: float(cfg[k]) for k in ('w_act', 'w_alpha', 'alpha_scale_deg', 'w_fb', 'fb_smooth', 'w_dist',
                                             'w_on_start', 'w_on_end', 'w_on_ramp_steps', 'w_psi', 'psi_scale_deg')}
        self.last_terms = None                      # 가중치를 곱한 항별 손실 텐서 (그래디언트 노름 기록용)

    def w_on(self, step):
        c = self.c
        if step is None or c['w_on_ramp_steps'] <= 0:
            return c['w_on_end']
        return c['w_on_start'] + (c['w_on_end'] - c['w_on_start']) * min(1.0, step / c['w_on_ramp_steps'])

    def forward(self, pred, target, dpsi=None, vkeep=None, step=None):
        c = self.c
        out = pred['out'].float()
        B = out.shape[0]
        keep = torch.ones(B, device=out.device) if vkeep is None else vkeep.float().to(out.device)
        act = perm_targets(target[:, :, :, 0])                     # (P, B, T, 3, C)
        cos = perm_targets(target[:, :, :, 1])
        sin = perm_targets(target[:, :, :, 2])
        dist = perm_targets(target[:, :, :, 3])
        ons = perm_targets(target[:, :, :, 4])
        alpha = torch.rad2deg(torch.asin(sin.clamp(-1.0, 1.0))) / c['alpha_scale_deg']
        logd = torch.log(dist.clamp(min=0.05))

        a_l = out[:, :, :, E_ACT][None]
        s_p = out[:, :, :, E_SIN][None]                            # α̂/20° (선형)
        f_l = out[:, :, :, E_FB][None]
        d_p = out[:, :, :, E_DIST][None]
        o_l = out[:, :, :, E_ON][None]
        km = keep[None, :, None, None, None]

        l_a = F.binary_cross_entropy_with_logits(a_l.expand_as(act), act, reduction='none')
        l_s = F.huber_loss(s_p.expand_as(alpha), alpha, reduction='none', delta=1.0)
        l_d = (logd - d_p).abs() / math.log(2.0)
        w_o = self.w_on(step)
        per = c['w_act'] * l_a + act * (c['w_alpha'] * l_s + c['w_dist'] * l_d)
        l_f = l_o = None
        if self.fb:
            eps = c['fb_smooth']
            tgt = torch.where(cos > 0, torch.full_like(cos, 1 - eps / 2), torch.full_like(cos, eps / 2))
            l_f = cos.abs() * F.binary_cross_entropy_with_logits(f_l.expand_as(cos), tgt, reduction='none')
            per = per + act * km * c['w_fb'] * l_f
        if self.on:
            l_o = F.binary_cross_entropy_with_logits(o_l.expand_as(ons), ons, reduction='none')
            per = per + act * km * w_o * l_o
        best = per.sum(dim=3).argmin(dim=0)                        # (B, T, C) 순열 선택

        def pick(x):
            idx = best[None, :, :, None, :].expand(1, *x.shape[1:])
            return torch.gather(x, 0, idx).squeeze(0)
        a_s = pick(act)
        a_k = a_s * keep[:, None, None, None]
        n_a, n_k = a_s.sum().clamp(min=1), a_k.sum().clamp(min=1)
        terms = dict(act=c['w_act'] * pick(l_a).mean(),
                     alpha=c['w_alpha'] * (pick(l_s) * a_s).sum() / n_a,
                     dist=c['w_dist'] * (pick(l_d) * a_s).sum() / n_a)
        raw = dict(act=terms['act'] / c['w_act'], alpha=terms['alpha'] / c['w_alpha'], dist=terms['dist'] / c['w_dist'])
        if l_f is not None:
            raw['fb'] = (pick(l_f) * a_k).sum() / n_k
            terms['fb'] = c['w_fb'] * raw['fb']
        if l_o is not None:
            raw['on'] = (pick(l_o) * a_k).sum() / n_k
            terms['on'] = w_o * raw['on']
        if self.rot and pred.get('dpsi') is not None and dpsi is not None:
            e = (((dpsi.float() - pred['dpsi'].float()) / c['psi_scale_deg']) ** 2).mean(dim=1)     # (B,)
            raw['rot'] = (e * keep).sum() / keep.sum().clamp(min=1)
            terms['rot'] = c['w_psi'] * raw['rot']
        loss = sum(terms.values())
        self.last_terms = terms
        parts = {k: float(v.detach()) for k, v in raw.items()}     # 가중치 곱하기 전 값
        parts['w_on'] = w_o
        parts['total'] = float(loss.detach())
        return loss, parts


def make_loss(cfg):
    return W1Loss(cfg) if cfg.get('loss_version', 'v0') == 'w1' else NADLoss(cfg)
