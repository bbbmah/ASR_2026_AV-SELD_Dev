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

    def forward(self, pred, target, dpsi=None):
        """pred: dict(out=(B,T,3,5,13), dpsi=(B,T)|None), target: (B,T,6,5,13), dpsi: (B,T) 도/프레임 정답.
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
