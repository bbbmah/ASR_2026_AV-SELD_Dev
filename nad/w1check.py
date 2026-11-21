"""
w1check.py -- MODEL_DEV_2.md '구현 전 확인' 2, 3번 (W1 프리셋으로 실행). 학습 없이 정답만 쓴다.

  2. 소프트 onscreen 라벨을 0.5 로 이진화했을 때 하드 라벨과 같은 비율 (train 클립 전체, 캐시를 만들며 잰다)
  3a. val 정답 θ -> (α, f) -> θ 복원이 원래 θ 와 같은지 (좌우 반전 포함), α 형식 오라클 채점 F_ext
  3b. 정답을 예측으로 넣었을 때 ②④⑥ 손실이 0, ③⑤ 는 스무딩과 소프트 라벨이 만드는 하한값인지
결과는 {run_dir}/w1_check.json 과 화면.
"""
import os
import json
import numpy as np
import pandas as pd
import torch

from .config import CLIP_FRAMES


def _H(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def roundtrip(cfg, log):
    from .prerender import extract_labels
    from .postproc import alpha_from_theta, theta_from_alpha
    ref = extract_labels(cfg, 'val', log)
    th = np.concatenate([pd.read_csv(os.path.join(ref, f))['azimuth'].values for f in sorted(os.listdir(ref))]).astype(float)
    out = {}
    for name, t in (('원래', th), ('좌우 반전', -th)):
        a = alpha_from_theta(t)
        front = np.cos(np.radians(t)) > 0
        back = theta_from_alpha(a, front)
        err = np.abs((back - t + 180.0) % 360.0 - 180.0)
        out[name] = dict(rows=int(len(t)), max_err_deg=float(err.max()), n_err_over_1e6=int((err > 1e-6).sum()))
        log(f'  θ->(α,f)->θ [{name}] {len(t)}행, 최대 차이 {err.max():.2e}°')
    return out


def loss_at_truth(cfg, log, n_clips=64, seed=0):
    """train(S0) 클립의 정답 텐서를 그대로 예측으로 넣어 W1 손실 항을 본다. 소프트 onscreen 캐시가 있어야 한다."""
    from .prerender import load_meta, load_onsoft
    from .data import events_to_adpit, dpsi_from_psi
    from .loss import W1Loss, perm_targets
    from .model import E_ACT, E_SIN, E_FB, E_DIST, E_ON
    key = 'train_s0'
    meta, soft = load_meta(cfg, key), load_onsoft(cfg, key)
    rng = np.random.default_rng(seed)
    names = sorted(meta)
    pick = rng.choice(len(names), n_clips, replace=False)
    Y, D = [], []
    for i in pick:
        n = names[i]
        flip = bool(rng.random() < 0.5)
        Y.append(events_to_adpit(meta[n]['ev'], flip, soft[n]))
        D.append(torch.from_numpy(dpsi_from_psi(meta[n]['psi']) * (-1 if flip else 1)))
    y, dpsi = torch.stack(Y), torch.stack(D)                       # (B, T, 6, 5, C), (B, T)
    eps = float(cfg['fb_smooth'])
    tr = {e: perm_targets(y[:, :, :, e])[0] for e in range(5)}     # 순열 0 (A0A0A0 + B112 + C345 = 값이 있는 그룹의 배치)
    act, cos, sin, dist, on = tr[0], tr[1], tr[2], tr[3], tr[4]
    out = torch.zeros(*act.shape[:3], 5, act.shape[-1])
    out[:, :, :, E_ACT] = torch.where(act > 0.5, 30.0, -30.0)
    out[:, :, :, E_SIN] = torch.rad2deg(torch.asin(sin.clamp(-1, 1))) / float(cfg['alpha_scale_deg'])
    fe = torch.where(cos > 0, 1 - eps / 2, eps / 2)
    out[:, :, :, E_FB] = torch.log(fe / (1 - fe))
    out[:, :, :, E_DIST] = torch.log(dist.clamp(min=0.05))
    oc = on.clamp(1e-6, 1 - 1e-6)
    out[:, :, :, E_ON] = torch.log(oc / (1 - oc))
    loss_fn = W1Loss(cfg)
    _, parts = loss_fn(dict(out=out, dpsi=dpsi.clone()), y, dpsi, step=None)
    a = act.numpy() > 0.5
    lb_fb = float((np.abs(cos.numpy()) * _H(fe.numpy()))[a].mean())
    lb_on = float(_H(on.numpy())[a].mean())
    res = dict(clips=int(n_clips), parts=parts, lower_bound=dict(fb=lb_fb, on=lb_on),
               soft_on_values_in_0_1_exclusive=float(((on.numpy() > 0.01) & (on.numpy() < 0.99))[a].mean()))
    log(f'  정답 입력 손실(가중치 전): ① {parts["act"]:.2e} ② {parts["alpha"]:.2e} ④ {parts["dist"]:.2e} ⑥ {parts.get("rot", float("nan")):.2e} | '
        f'③ {parts["fb"]:.5f} (하한 {lb_fb:.5f}) ⑤ {parts["on"]:.5f} (하한 {lb_on:.5f})')
    return res


def run(cfg, log=print):
    from .prerender import build_onsoft, build_meta
    from .runner import oracle_check, atomic_json
    assert cfg['loss_version'] == 'w1', 'W1 프리셋(N2w)으로 실행하세요'
    res = {}
    log('[확인 2] 소프트 onscreen 캐시 (이진화 일치율은 아래 로그)')
    for k in ('train_s0', 'train_s1'):
        build_meta(cfg, k, log)
        build_onsoft(cfg, k, log)
    log('[확인 3a] θ 복원')
    res['roundtrip'] = roundtrip(cfg, log)
    r = oracle_check(cfg, log=log)
    res['oracle_alpha'] = dict(F_ext=r['ext']['F'], F_basic=r['basic']['F'], DOA_ext=r['ext']['DOA_err'])
    log('[확인 3b] 정답 입력 손실')
    res['loss_at_truth'] = loss_at_truth(cfg, log)
    atomic_json(os.path.join(cfg['run_dir'], 'w1_check.json'), res)
    return res
