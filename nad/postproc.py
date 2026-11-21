"""
postproc.py -- 추론 후처리 (CLAUDE.md 3.4)

1. 활성 a > 0.5 인 칸만. θ = atan2(s, ±√(1-s²)), f ≥ 0.5 면 +(앞), 아니면 -(뒤).
   (앞/뒤 손실을 끈 1단계 모델은 f 를 학습하지 않았으므로 항상 앞(+)으로 둔다: fb_trained=False)
2. o > τ 면 앞(+)으로 고정.
3. 검출 위치 보정(켜고 끄기): o > τ 이고, 같은 클래스의 OWL-ViT 열 점수(시그모이드)가 기준 이상인 열이 예측 θ 의 ±창 안에 있으면
   θ 를 그 열(점수 최대)의 방위각으로 바꾼다.
4. 트랙 통합: 같은 클래스 트랙끼리 원형 방위각 차이가 20° 미만이면 평균한다(방향은 원형 평균, 거리와 o 는 평균).
5. 거리 = exp(ℓ) 미터 -> 베이스라인 write_to_dcase_output_format 이 round(×100) cm 로 쓴다. onscreen 은 o > 0.5.
"""
import os
import numpy as np

from .model import E_ACT, E_SIN, E_FB, E_DIST, E_ON


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def circ_diff(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def decode_clip(out, pp, video=None, col_az_deg=None, fb_trained=True, on_trained=True, snap_log=None):
    """out: (T, 3, 5, 13) numpy (활성 함수 전), video: (T, 24, 77) 또는 None.
    pp['onscreen_front'](없으면 True): False 면 2번(o > τ 면 앞 고정)을 끈다(진단용).
    snap_log: 리스트를 주면 3번이 θ 를 바꾼 칸마다 (프레임, 클래스, 원래 θ, 바뀐 θ) 를 덧붙인다(진단용).
    반환: {frame: [[class, 0, az_deg, dist_m, onscreen(0/1)], ...]}"""
    T, K, _, C = out.shape
    a = _sig(out[:, :, E_ACT]) > pp['act_thresh']                              # (T, 3, C)
    s = np.clip(np.tanh(out[:, :, E_SIN]), -1.0, 1.0)
    front = _sig(out[:, :, E_FB]) >= 0.5 if fb_trained else np.ones_like(a)
    o = _sig(out[:, :, E_ON]) if on_trained else np.zeros(a.shape)
    tau = pp['onscreen_tau']
    if pp.get('onscreen_front', True):
        front = front | (o > tau)
    c = np.sqrt(np.clip(1.0 - s ** 2, 0.0, 1.0))
    th = np.degrees(np.arctan2(s, np.where(front, c, -c)))                     # (T, 3, C)
    dist = np.exp(out[:, :, E_DIST])
    if pp.get('snap_enable') and video is not None and col_az_deg is not None:
        col = np.asarray(col_az_deg)                                           # (24,)
        score = _sig(video[:, :, :C].astype(np.float32))                       # (T, 24, C)
        cand = score >= pp['snap_score']
        for t, k, ci in zip(*np.nonzero(a & (o > tau))):
            ok = cand[t, :, ci] & (circ_diff(col, th[t, k, ci]) <= pp['snap_window_deg'])
            if ok.any():
                j = np.argmax(np.where(ok, score[t, :, ci], -1.0))
                if snap_log is not None:
                    snap_log.append((int(t), int(ci), float(th[t, k, ci]), float(col[j])))
                th[t, k, ci] = col[j]
    res = {}
    md = pp['merge_deg']
    for t, ci in zip(*np.nonzero(a.any(axis=1))):
        tr = [[th[t, k, ci], dist[t, k, ci], o[t, k, ci]] for k in range(K) if a[t, k, ci]]
        merged = True
        while merged and len(tr) > 1:                                          # 가장 가까운 쌍부터 합친다
            merged = False
            best = None
            for i in range(len(tr)):
                for j in range(i + 1, len(tr)):
                    dd = circ_diff(tr[i][0], tr[j][0])
                    if dd < md and (best is None or dd < best[0]):
                        best = (dd, i, j)
            if best is not None:
                _, i, j = best
                n = tr[i][3] if len(tr[i]) > 3 else 1
                m = tr[j][3] if len(tr[j]) > 3 else 1
                v = n * np.exp(1j * np.radians(tr[i][0])) + m * np.exp(1j * np.radians(tr[j][0]))
                new = [float(np.degrees(np.angle(v))), (n * tr[i][1] + m * tr[j][1]) / (n + m), (n * tr[i][2] + m * tr[j][2]) / (n + m), n + m]
                tr = [x for q, x in enumerate(tr) if q not in (i, j)] + [new]
                merged = True
        res.setdefault(int(t), [])
        for x in tr:
            az = (x[0] + 180.0) % 360.0 - 180.0
            res[int(t)].append([int(ci), 0, float(az), float(x[1]), int(x[2] > 0.5)])
    return res


def write_clip(out_dict, out_dir, name):
    """베이스라인 utils.write_to_dcase_output_format 을 그대로 쓴다 (값: [class, source, az, dist_m, onscreen])."""
    import utils                                     # for_dataset/baseline/seld_challenge/utils.py (config.add_seld_path)
    utils.write_to_dcase_output_format(out_dict, out_dir, name + '.csv')
