"""
errdiag.py -- 오라클 교체와 오류 분해. 기존 예측 csv(runs/<run>/pred_<split>[_last]/)만 읽는다. 추론, 학습 없음.

결과: {nad_dir}/diag/ERR_01/<run>_<split>[_last].json, tables.md. 고친 예측 csv 는 {local_root}/tmp/err01/ 에만 쓴다.
채점은 베이스라인 score.score_dirs 그대로다(lad_req_onscreen=False 이므로 onscreen 은 F 에 들어가지 않는다).

1. 오라클 교체 (같은 프레임·같은 클래스에서 채점과 같은 헝가리안 매칭으로 짝지은 예측만 고친다. 짝 없는 예측, 놓친 정답은 그대로)
   fb    접은 매칭 짝의 앞/뒤만 정답 쪽으로 (예측 θ 를 180°−θ 로 뒤집어 접은 값은 유지). 정답 cos=0 이면 그대로
   on    접지 않은 매칭(ext) 짝의 onscreen 을 정답으로 -> OSA 만 바뀐다
   fb_on fb 뒤에 on
   doa   접은 매칭 짝의 θ 를 정답 θ 로 (거리는 그대로)
   loc   접은 매칭 짝의 θ, 거리를 정답으로 -> 남는 오류는 검출(개수, 클래스)뿐
2. 오류 분해 (ext 채점과 같은 셈. 프레임·클래스마다)
   정답 R 개, 예측 P 개:
     둘 다 있음  K=min(P,R) 짝(접지 않은 매칭). 짝이 20° 넘으면: 같은 짝의 접은 오차 ≤20° 이면 '앞뒤', 아니면 '각도'.
                 20° 이내인데 상대 거리 오차 >1 이면 '거리'. R>P 남는 정답 = 놓침(FN), P>R 남는 예측 = 남는 트랙(FP)
     정답만 있음 놓침(FN),  예측만 있음 헛검출(FP)
   클래스 혼동: 한 프레임에서 짝 없는 정답과 짝 없는 다른 클래스 예측을 접은 방위각 ≤20° 로 헝가리안 매칭한 것.
                이 짝의 FN 은 '혼동 FN', FP 는 '혼동 FP' 로 따로 센다(나머지 FN = '놓침', FP = '헛검출'/'남는 트랙').
   '고치면 F_ext': 한 오류 종류만 없앴을 때의 macro F_ext (FN/공간 오류 -> TP, FP -> 없앰). 다른 오류는 그대로.
   폴리포니: 프레임의 정답 개수(1, 2, 3+)별 정답 중 같은 클래스 예측과 짝지어진 비율, ext TP 비율.
   이벤트: 정답 (클립, 클래스, source) 를 한 이벤트로 보고, 그 프레임 중 같은 클래스 예측이 있는 비율(덮인 비율).
           덮인 비율 0 인 이벤트 = 통째로 놓친 이벤트. 예측 (클립, 클래스, source) 중 정답 같은 클래스와 한 프레임도
           겹치지 않는 것 = 통째로 헛검출한 예측 트랙.
"""
import os
import json
import shutil
import numpy as np

from .config import add_seld_path, CLIP_FRAMES
from .runner import modality

NC = 13
THR, RD_THR = 20.0, 1.0
ORACLES = ('fb', 'on', 'fb_on', 'doa', 'loc')
ERR_ID = 'ERR_01'
CATS = ('fb', 'ang', 'dist', 'fn_miss', 'conf', 'fp_none', 'fp_extra')
CAT_KO = dict(fb='앞뒤', ang='각도', dist='거리', fn_miss='놓침', conf='클래스 혼동', fp_none='헛검출', fp_extra='남는 트랙')


def out_dir(cfg):
    return os.path.join(cfg['nad_dir'], 'diag', ERR_ID)


def tmp_dir(cfg, *a):
    return os.path.join(cfg['local_root'], 'tmp', ERR_ID.lower().replace('_', ''), *a)


def _wrap(a):
    return (np.asarray(a, dtype=float) + 180) % 360 - 180


def _mirror(a):
    return _wrap(180 - np.asarray(a, dtype=float))


def read_rows(path):
    """csv -> (N, 6) float 배열 frame, class, source, az, dist, on (채점 범위 밖 프레임은 뺀다)"""
    if not os.path.exists(path):
        return np.zeros((0, 6))
    x = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    x = x.reshape(-1, 6) if x.size else np.zeros((0, 6))
    return x[(x[:, 0] >= 0) & (x[:, 0] < CLIP_FRAMES)]


def write_rows(path, x):
    with open(path, 'w') as f:
        f.write('frame,class,source,azimuth,distance,onscreen\n')
        for r in x:
            f.write(f'{int(r[0])},{int(r[1])},{int(r[2])},{r[3]:g},{r[4]:g},{int(r[5])}\n')


def _groups(x):
    """(frame, class) -> 행 번호 배열"""
    g = {}
    for i, (t, c) in enumerate(zip(x[:, 0].astype(int), x[:, 1].astype(int))):
        g.setdefault((t, c), []).append(i)
    return {k: np.array(v) for k, v in g.items()}


def _match(gaz, paz, fold):
    from seld_label_utils import least_distance_between_gt_pred
    cost, ri, ci = least_distance_between_gt_pred(np.asarray(gaz, float), np.asarray(paz, float), fold=fold)
    return cost, ri, ci


def _fold_diff(a, b):
    from seld_label_utils import az_difference
    return az_difference(a, b, True)


def oracle_rows(g, p, kind):
    """kind 오라클로 고친 예측 행"""
    q = p.copy()
    G, P = _groups(g), _groups(p)
    if kind in ('fb', 'fb_on', 'doa', 'loc'):
        for k, pi in P.items():
            gi = G.get(k)
            if gi is None:
                continue
            _, ri, ci = _match(g[gi, 3], q[pi, 3], True)
            for r, c in zip(ri, ci):
                a, b = gi[r], pi[c]
                if kind in ('fb', 'fb_on'):
                    gc, pc = np.cos(np.radians(g[a, 3])), np.cos(np.radians(q[b, 3]))
                    if abs(gc) > 1e-9 and (gc > 0) != (pc >= 0):
                        q[b, 3] = _mirror(q[b, 3])
                else:
                    q[b, 3] = g[a, 3]
                    if kind == 'loc':
                        q[b, 4] = g[a, 4]
    if kind in ('on', 'fb_on'):
        for k, pi in P.items():
            gi = G.get(k)
            if gi is None:
                continue
            _, ri, ci = _match(g[gi, 3], q[pi, 3], False)
            for r, c in zip(ri, ci):
                q[pi[c], 5] = g[gi[r], 5]
    return q


class Acc:
    """오류 분해 누적"""
    def __init__(self):
        z = lambda: np.zeros(NC)
        self.nref, self.npred, self.tp = z(), z(), z()
        self.c = {k: z() for k in CATS}            # conf 는 FN 쪽 개수(혼동 FP 는 conf_fp)
        self.conf_fp = z()
        self.conf_mat = np.zeros((NC, NC), dtype=int)
        self.poly = np.zeros((3, 3))               # [폴리포니 1, 2, 3+] x [정답 수, 짝지어진 수, ext TP 수]
        self.ev_len, self.ev_cov = [[] for _ in range(NC)], [[] for _ in range(NC)]
        self.fa_tracks, self.pred_tracks = z(), z()

    def clip(self, g, p):
        G, P = _groups(g), _groups(p)
        frames = set(k[0] for k in G) | set(k[0] for k in P)
        for t in frames:
            ug, up = [], []          # 짝 없는 정답 (class, az), 짝 없는 예측 (class, az, 종류)
            n_t = sum(len(G[k]) for k in G if k[0] == t)
            pi_ = min(n_t, 3) - 1
            for c in range(NC):
                gi, pi = G.get((t, c)), P.get((t, c))
                if gi is not None:
                    self.nref[c] += len(gi)
                    self.poly[pi_, 0] += len(gi)
                if pi is not None:
                    self.npred[c] += len(pi)
                if gi is not None and pi is not None:
                    cost, ri, ci = _match(g[gi, 3], p[pi, 3], False)
                    fold = _fold_diff(g[gi[ri], 3], p[pi[ci], 3])
                    rd = np.abs(g[gi[ri], 4] - p[pi[ci], 4]) / (g[gi[ri], 4] + np.finfo(float).eps)
                    bad_ang = cost > THR
                    fb = bad_ang & (fold <= THR)
                    ang = bad_ang & (fold > THR)
                    dist = (~bad_ang) & (rd > RD_THR)
                    tp = int((~bad_ang & ~dist).sum())
                    self.tp[c] += tp
                    self.c['fb'][c] += fb.sum(); self.c['ang'][c] += ang.sum(); self.c['dist'][c] += dist.sum()
                    self.poly[pi_, 1] += len(ri); self.poly[pi_, 2] += tp
                    ug += [(c, g[gi[r], 3]) for r in set(range(len(gi))) - set(ri)]
                    up += [(c, p[pi[j], 3], 'fp_extra') for j in set(range(len(pi))) - set(ci)]
                elif gi is not None:
                    ug += [(c, g[i, 3]) for i in gi]
                elif pi is not None:
                    up += [(c, p[i, 3], 'fp_none') for i in pi]
            conf_g, conf_p = set(), set()
            if ug and up:
                from scipy.optimize import linear_sum_assignment
                cm = np.full((len(ug), len(up)), 1e6)
                for a, (cg, ag) in enumerate(ug):
                    for b, (cp, ap, _) in enumerate(up):
                        if cg != cp:
                            d = float(_fold_diff(np.array([ag]), np.array([ap]))[0])
                            if d <= THR:
                                cm[a, b] = d
                ri, ci = linear_sum_assignment(cm)
                for a, b in zip(ri, ci):
                    if cm[a, b] < 1e6:
                        conf_g.add(a); conf_p.add(b)
                        self.conf_mat[ug[a][0], up[b][0]] += 1
            for a, (cg, _) in enumerate(ug):
                self.c['conf' if a in conf_g else 'fn_miss'][cg] += 1
            for b, (cp, _, kind) in enumerate(up):
                if b in conf_p:
                    self.conf_fp[cp] += 1
                else:
                    self.c[kind][cp] += 1
        # 이벤트
        pf = {}
        for r in p:
            pf.setdefault(int(r[1]), set()).add(int(r[0]))
        gf = {}
        for r in g:
            gf.setdefault(int(r[1]), set()).add(int(r[0]))
        ev = {}
        for r in g:
            ev.setdefault((int(r[1]), int(r[2])), set()).add(int(r[0]))
        for (c, _), fr in ev.items():
            self.ev_len[c].append(len(fr))
            self.ev_cov[c].append(len(fr & pf.get(c, set())) / len(fr))
        pt = {}
        for r in p:
            pt.setdefault((int(r[1]), int(r[2])), set()).add(int(r[0]))
        for (c, _), fr in pt.items():
            self.pred_tracks[c] += 1
            if not (fr & gf.get(c, set())):
                self.fa_tracks[c] += 1

    def f_ext(self, fix=None):
        """macro F_ext (정답 있는 클래스 평균). fix = 없앨 오류 종류"""
        tp = self.tp.copy()
        L = self.c['fb'] + self.c['ang'] + self.c['dist']
        fn = self.c['fn_miss'] + self.c['conf']
        fp = self.c['fp_none'] + self.c['fp_extra'] + self.conf_fp
        if fix in ('fb', 'ang', 'dist'):
            tp, L = tp + self.c[fix], L - self.c[fix]
        elif fix == 'fn_miss':
            tp, fn = tp + self.c['fn_miss'], fn - self.c['fn_miss']
        elif fix == 'conf':
            tp, fn, fp = tp + self.c['conf'], fn - self.c['conf'], fp - self.conf_fp
        elif fix in ('fp_none', 'fp_extra'):
            fp = fp - self.c[fix]
        F = tp / (np.finfo(float).eps + tp + L + 0.5 * (fp + fn))
        m = self.nref > 0
        return float(F[m].mean()), F

    def summary(self):
        F0, Fc = self.f_ext()
        ev_n = np.array([len(v) for v in self.ev_cov])
        ev_zero = np.array([int(np.sum(np.array(v) == 0)) for v in self.ev_cov])
        ev_cov = [float(np.mean(v)) if v else None for v in self.ev_cov]
        ev_len_zero = [float(np.mean(np.array(l)[np.array(v) == 0])) if np.any(np.array(v) == 0) else None
                       for l, v in zip(self.ev_len, self.ev_cov)]
        return dict(F_ext=F0, F_ext_class=Fc.tolist(),
                    fix={k: self.f_ext(k)[0] for k in CATS},
                    fix_class={k: self.f_ext(k)[1].tolist() for k in CATS},
                    nref=self.nref.tolist(), npred=self.npred.tolist(), tp=self.tp.tolist(),
                    counts={k: v.tolist() for k, v in self.c.items()}, conf_fp=self.conf_fp.tolist(),
                    conf_mat=self.conf_mat.tolist(),
                    poly=dict(n=self.poly[:, 0].tolist(), matched=self.poly[:, 1].tolist(), tp=self.poly[:, 2].tolist()),
                    events=dict(n=ev_n.tolist(), zero=ev_zero.tolist(), mean_cov=ev_cov, len_zero=ev_len_zero,
                                len_all=[float(np.mean(l)) if l else None for l in self.ev_len]),
                    pred_tracks=self.pred_tracks.tolist(), fa_tracks=self.fa_tracks.tolist())


def _score(cfg, pred_dir, ref_dir):
    add_seld_path(cfg)
    from score import score_dirs
    r = score_dirs(pred_dir, ref_dir, CLIP_FRAMES, modality(cfg), None, exclude_absent_classes=True)
    return dict(F_ext=r['ext']['F'], F_basic=r['basic']['F'], DOAE_ext=r['ext']['DOA_err'], DOAE_basic=r['basic']['DOA_err'],
                RDE=r['ext']['rel_dist_err'], OSA=r['ext']['onscreen_acc'], F_ext_class=r['ext']['classwise']['F'])


def run_one(cfg, run, split, suf, ref_dir, log):
    add_seld_path(cfg)
    pred = os.path.join(cfg['out_root'], run, f'pred_{split}{suf}')
    if not os.path.isdir(pred):
        raise SystemExit(f'[중단] 예측 폴더가 없습니다: {pred}')
    names = sorted(f for f in os.listdir(ref_dir) if f.endswith('.csv'))
    res = dict(run=run, split=split, ckpt=suf.strip('_') or 'best', pred_dir=pred, oracle={})
    res['oracle']['orig'] = o = _score(cfg, pred, ref_dir)
    sj = os.path.join(cfg['out_root'], run, f'scores_{split}{suf}.json')
    if os.path.exists(sj):
        s = json.load(open(sj))['scores']
        res['saved_F_ext'] = s['ext']['F']
        if abs(s['ext']['F'] - o['F_ext']) > 1e-6:
            log(f'  [주의] {run} {split}{suf}: 다시 채점한 F_ext {o["F_ext"]:.5f} != 저장된 값 {s["ext"]["F"]:.5f}')
    acc = Acc()
    dirs = {k: tmp_dir(cfg, f'{run}_{split}{suf}', k) for k in ORACLES}
    for d in dirs.values():
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    for n in names:
        g, p = read_rows(os.path.join(ref_dir, n)), read_rows(os.path.join(pred, n))
        acc.clip(g, p)
        for k, d in dirs.items():
            write_rows(os.path.join(d, n), oracle_rows(g, p, k))
    for k, d in dirs.items():
        res['oracle'][k] = _score(cfg, d, ref_dir)
    res['err'] = acc.summary()
    diff = abs(res['err']['F_ext'] - o['F_ext'])
    res['err']['check_F_ext_diff'] = diff
    if diff > 1e-6:
        log(f'  [주의] 오류 분해로 다시 센 F_ext {res["err"]["F_ext"]:.5f} != 채점 {o["F_ext"]:.5f}')
    log(f'  {run} {split}{suf}: F_ext {o["F_ext"]*100:.2f} | 오라클 fb {res["oracle"]["fb"]["F_ext"]*100:.2f}, '
        f'doa {res["oracle"]["doa"]["F_ext"]*100:.2f}, loc {res["oracle"]["loc"]["F_ext"]*100:.2f} | 분해 확인 차이 {diff:.1e}')
    return res


def run_all(cfg, runs, splits, last, log):
    from .prerender import extract_labels
    os.makedirs(out_dir(cfg), exist_ok=True)
    for split in splits:
        ref = extract_labels(cfg, split, log)
        for run in runs:
            for suf in ([''] + (['_last'] if last else [])):
                r = run_one(cfg, run, split, suf, ref, log)
                p = os.path.join(out_dir(cfg), f'{run}_{split}{suf}.json')
                json.dump(r, open(p, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
                log(f'  저장: {p}')


# ====================================================================== 표
def _p(v):
    return '–' if v is None else f'{v * 100:.2f}'


def write_tables(cfg, log=print):
    d = out_dir(cfg)
    R = []
    for f in sorted(os.listdir(d)):
        if f.endswith('.json'):
            R.append(json.load(open(os.path.join(d, f), encoding='utf8')))
    R.sort(key=lambda r: (r['split'] != 'val', r['ckpt'] != 'best', r['run']))
    L = [f'# {ERR_ID} 결과 표 (오라클 교체, 오류 분해)', '',
         '설명은 `nad/errdiag.py` 머리말. F 는 %, 정답 있는 클래스 macro 평균. onscreen 은 F 에 들어가지 않는다(OSA 만).', '']
    L += ['## 1. 오라클 교체', '',
          '| split | run | ckpt | F_ext | F_basic | OSA | fb: F_ext | on: OSA | fb_on: F_ext / OSA | doa: F_ext / F_basic | loc: F_ext |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in R:
        o = r['oracle']
        L.append(f"| {r['split']} | {r['run']} | {r['ckpt']} | {_p(o['orig']['F_ext'])} | {_p(o['orig']['F_basic'])} | {_p(o['orig']['OSA'])} | "
                 f"{_p(o['fb']['F_ext'])} | {_p(o['on']['OSA'])} | {_p(o['fb_on']['F_ext'])} / {_p(o['fb_on']['OSA'])} | "
                 f"{_p(o['doa']['F_ext'])} / {_p(o['doa']['F_basic'])} | {_p(o['loc']['F_ext'])} |")
    L += ['', '## 2. 한 종류만 고쳤을 때 F_ext (오류 분해 셈)', '',
          '| split | run | ckpt | 지금 | ' + ' | '.join(CAT_KO[k] for k in CATS) + ' |',
          '|---|---|---|---|' + '---|' * len(CATS)]
    for r in R:
        e = r['err']
        L.append(f"| {r['split']} | {r['run']} | {r['ckpt']} | {_p(e['F_ext'])} | " + ' | '.join(f"+{(e['fix'][k] - e['F_ext']) * 100:.2f}" for k in CATS) + ' |')
    L += ['', '## 3. 오류 개수 (전체 클래스 합, 정답 수 대비 %)', '',
          '| split | run | ckpt | 정답 수 | TP | ' + ' | '.join(CAT_KO[k] for k in CATS) + ' | 혼동 FP |',
          '|---|---|---|---|---|' + '---|' * len(CATS) + '---|']
    for r in R:
        e = r['err']; n = sum(e['nref'])
        L.append(f"| {r['split']} | {r['run']} | {r['ckpt']} | {int(n)} | {sum(e['tp']) / n * 100:.1f} | " +
                 ' | '.join(f"{sum(e['counts'][k]) / n * 100:.1f}" for k in CATS) + f" | {sum(e['conf_fp']) / n * 100:.1f} |")
    L += ['', '## 4. 폴리포니별 (정답 중 짝지어진 비율 / ext TP 비율, %)', '',
          '| split | run | ckpt | 1개 | 2개 | 3개 이상 |', '|---|---|---|---|---|---|']
    for r in R:
        pl = r['err']['poly']
        cells = [f"{m / n * 100:.1f} / {t / n * 100:.1f} (n={int(n)})" if n else '–' for n, m, t in zip(pl['n'], pl['matched'], pl['tp'])]
        L.append(f"| {r['split']} | {r['run']} | {r['ckpt']} | " + ' | '.join(cells) + ' |')
    for r in R:
        e = r['err']; ev = e['events']
        L += ['', f"## 5. 클래스별: {r['split']} {r['run']} {r['ckpt']}", '',
              '오류 칸은 그 클래스 정답 수 대비 %(FP 종류도 정답 수로 나눔, 정답 0 이면 괄호 안 개수). 고치면 최대 이득 = 그 클래스 F_ext 증가(%p).', '',
              '| 클래스 | 정답 | 예측 | F_ext | ' + ' | '.join(CAT_KO[k] for k in CATS) + ' | 고치면 최대 이득(종류) | 이벤트 수 | 통째로 놓친 이벤트 | 덮인 비율 평균 | 헛검출 예측 트랙 |',
              '|---|---|---|---|' + '---|' * len(CATS) + '---|---|---|---|---|']
        for c in range(NC):
            n = e['nref'][c]
            if n == 0 and e['npred'][c] == 0:
                continue
            gains = {k: e['fix_class'][k][c] - e['F_ext_class'][c] for k in CATS}
            kb = max(gains, key=gains.get)
            pct = lambda k: f"{e['counts'][k][c] / n * 100:.0f}" if n else f"({int(e['counts'][k][c])})"
            cov = ev['mean_cov'][c]
            cov = '–' if cov is None else f'{cov * 100:.0f}'
            L.append(f"| {c} | {int(n)} | {int(e['npred'][c])} | {e['F_ext_class'][c] * 100:.1f} | " + ' | '.join(pct(k) for k in CATS) +
                     f" | {CAT_KO[kb]} +{gains[kb] * 100:.0f} | {ev['n'][c]} | {ev['zero'][c]} | {cov} | "
                     f"{int(e['fa_tracks'][c])}/{int(e['pred_tracks'][c])} |")
        cm = np.array(e['conf_mat'])
        top = sorted(((cm[i, j], i, j) for i in range(NC) for j in range(NC) if cm[i, j] > 0), reverse=True)[:8]
        if top:
            L.append('')
            L.append('클래스 혼동 상위 (정답 -> 예측: 프레임 수): ' + ', '.join(f'{i}->{j}: {v}' for v, i, j in top))
    p = os.path.join(d, 'tables.md')
    open(p, 'w', encoding='utf8').write('\n'.join(L) + '\n')
    log(f'저장: {p}')
