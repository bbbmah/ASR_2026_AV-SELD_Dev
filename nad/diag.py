"""
diag.py -- 1차 결과 진단 (DIGNOSIS_1.md). 기존 체크포인트로 추론만 한다. 학습, 하이퍼파라미터 탐색 없음.

결과는 {nad_dir}/diag/DIAG_01/ 에 새로 쓴다(n2.json, n1.json, tables.md). 기존 runs/ 파일은 읽기만 한다.
예측 csv 는 코랩 로컬 {local_root}/tmp/diag01/ 에만 둔다.

 1. N2 후처리 4조합 (검출 위치 보정 켬/끔 × onscreen→앞 고정 켬/끔) + 보정이 θ 를 바꾼 비율과 정답에 가까워진 비율
 2. N2 영상 원래 / 0 / 섞기 (1번에서 val F_ext 최고 조합으로)
    - 0: 표준화 뒤 0 이 되게 원값을 train 평균으로 넣는다((C) 열 토큰, (D) 회전 특징 모두 같은 입력을 쓴다).
         후처리에는 영상 점수를 주지 않는다(보정 없음).
    - 섞기: 베이스라인 run.apply_eval_video 와 같은 규칙(시드 1000+seed 순열, 순서상 다음 클립의 영상, 자기 영상 없음).
         후처리 보정도 섞인 영상의 점수를 쓴다.
 3. 앞/뒤 정확도: 같은 프레임·같은 클래스에서 접은 방위각으로 헝가리안 매칭(채점 basic 과 같은 매칭)한 쌍.
    정답 cos 부호와 예측 cos 부호가 같은 비율. 정답 cos=0(±90°)인 쌍은 뺀다.
 4. onscreen: 채점 ext 와 같은 매칭(접지 않은 방위각)으로 짝지은 쌍에서 정밀도/재현율/F1, 전부 0 으로 냈을 때 OSA(재채점).
 5. N1-pre vs N1-otf: 기존 예측 csv 로 클립별 집계값을 만들고 같은 클립을 뽑는 쌍 부트스트랩(1000번).
    for_dataset/build/eval_all.py(vec), analyze_results.py(metrics) 와 같은 계산이다. 두 파일은 import 할 때
    로컬 경로를 읽어서 그대로 import 할 수 없으므로 같은 식을 옮겨 썼다.
"""
import os
import json
import shutil
import time
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import add_seld_path, CLIP_FRAMES
from .model import E_ACT, E_FB, E_ON
from .postproc import decode_clip, write_clip, circ_diff
from . import runner as R

SPLITS = ('val', 'test')
COMBOS = dict(a=(True, True), b=(False, True), c=(True, False), d=(False, False))   # (검출 위치 보정, onscreen→앞 고정)
BOOT_B, BOOT_SEED = 1000, 20261006
FB_BASELINE = 2 / 3          # 이보다 낮으면 앞/뒤를 둘 다 내는 쪽이 점수상 유리 (DIGNOSIS_1.md 3번)
EPS = np.finfo(float).eps


DIAG_ID = 'DIAG_01'          # diag_run.py --diag 로 바꾼다 (W1 진단은 다른 폴더에 써서 기존 결과를 덮지 않는다)


def out_dir(cfg):
    return os.path.join(cfg['nad_dir'], 'diag', DIAG_ID)


def tmp_dir(cfg, *a):
    return os.path.join(cfg['local_root'], 'tmp', DIAG_ID.lower().replace('_', ''), *a)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def summarize(r):
    return dict(F_ext=r['ext']['F'], F_basic=r['basic']['F'], DOAE_ext=r['ext']['DOA_err'], RDE=r['ext']['rel_dist_err'],
                OSA=r['ext']['onscreen_acc'])


# ====================================================================== 추론
class VideoMode(Dataset):
    """평가 클립의 영상 특징을 real | zeros | shuffle 로 바꿔 준다."""
    def __init__(self, base, mode, seed, vis_mean):
        self.base, self.mode = base, mode
        n = len(base)
        self.src = np.arange(n)
        if mode == 'shuffle':
            order = np.random.default_rng(1000 + seed).permutation(n)
            src = np.empty(n, dtype=np.int64)
            src[order] = np.roll(order, -1)
            assert (src != np.arange(n)).all()
            self.src = src
        self.vmean = vis_mean.float()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = self.base[i]
        if self.mode == 'zeros':
            d['video'] = self.vmean.expand(*d['video'].shape).clone()
        elif self.mode == 'shuffle':
            d['video'] = self.base.video.get(self.base.names[self.src[i]], False)
        return d


@torch.no_grad()
def forward_all(model, ds, cfg, dev, keep_video):
    """반환: [(이름, 출력 (T,3,5,13) float32, 후처리용 영상 (T,24,77) float16 또는 None)]"""
    model.eval()
    kept = []
    for b in R.loader(ds, cfg):
        b = R.to_dev(b, dev)
        with torch.autocast('cuda', dtype=R.amp_dtype(), enabled=cfg['amp'] and dev == 'cuda'):
            pred = model(b['audio'], b['clap'], b['video'])
        out = pred['out'].float().cpu().numpy()
        vid = b['video'].float().cpu().numpy().astype(np.float16) if keep_video else None
        for i, n in enumerate(b['name']):
            kept.append((n, out[i], None if vid is None else vid[i]))
    return kept


def decode_write(kept, pp, col, d, cfg, snap_log=None):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for n, out, vid in kept:
        lg = [] if snap_log is not None else None
        write_clip(decode_clip(out, pp, None if vid is None else vid.astype(np.float32), col, cfg['loss_fb'], cfg['loss_onscreen'], lg), d, n)
        if snap_log is not None:
            snap_log[n] = lg


# ====================================================================== 항목별 계산
def snap_stats(snap_log, ref_dir, kept, thr):
    from seld_label_utils import load_labels
    n_active = int(sum((_sig(out[:, :, E_ACT]) > thr).sum() for _, out, _ in kept))
    n = closer = worse = same = nogt = 0
    for name, ent in snap_log.items():
        if not ent:
            continue
        gt = load_labels(os.path.join(ref_dir, name + '.csv'), convert_to_cartesian=False)
        for t, c, old, new in ent:
            n += 1
            az = np.array([r[2] for r in gt.get(t, []) if r[0] == c], dtype=float)
            if az.size == 0:
                nogt += 1
                continue
            do, dn = circ_diff(az, old).min(), circ_diff(az, new).min()
            if dn < do - 1e-6:
                closer += 1
            elif dn > do + 1e-6:
                worse += 1
            else:
                same += 1
    wg = n - nogt
    return dict(n_active=n_active, n_changed=n, changed_ratio=n / max(n_active, 1), closer=closer, worse=worse, same=same, no_gt=nogt,
                closer_ratio=closer / max(n, 1), closer_ratio_with_gt=closer / max(wg, 1))


def matched_pairs(pred_dir, ref_dir, fold):
    """같은 프레임·클래스에서 채점과 같은 헝가리안 매칭. 반환 (K, 5): 정답 θ, 예측 θ, 정답 on, 예측 on, (접은/접지 않은) 오차"""
    from seld_label_utils import load_labels, organize_labels, least_distance_between_gt_pred
    rows = []
    for f in sorted(os.listdir(ref_dir)):
        if not f.endswith('.csv'):
            continue
        gt = organize_labels(load_labels(os.path.join(ref_dir, f), convert_to_cartesian=False), CLIP_FRAMES)
        pp = os.path.join(pred_dir, f)
        pr = organize_labels(load_labels(pp, convert_to_cartesian=False) if os.path.exists(pp) else {}, CLIP_FRAMES)
        for t in range(CLIP_FRAMES):
            for c in gt[t]:
                if c not in pr[t]:
                    continue
                g = np.array(list(gt[t][c].values()), dtype=float)
                p = np.array(list(pr[t][c].values()), dtype=float)
                cost, ri, ci = least_distance_between_gt_pred(g[:, 0], p[:, 0], fold=fold)
                for k in range(len(ri)):
                    rows.append((g[ri[k], 0], p[ci[k], 0], g[ri[k], 2], p[ci[k], 2], cost[k]))
    return np.array(rows, dtype=float).reshape(-1, 5)


def fb_table(P):
    """앞/뒤 정확도. 그룹마다 전체 매칭 쌍과, 접은 오차 ≤ 20° 쌍(앞/뒤만 맞으면 TP 가 되는 쌍)."""
    gc, pc = np.cos(np.radians(P[:, 0])), np.cos(np.radians(P[:, 1]))
    keep = np.abs(gc) > 1e-6
    ok = (gc > 0) == (pc >= 0)
    near = P[:, 4] <= 20
    a = np.abs((P[:, 0] + 180) % 360 - 180)
    groups = [('전체', keep), ('정답 onscreen=1', keep & (P[:, 2] > 0.5)), ('정답 onscreen=0', keep & (P[:, 2] < 0.5))]
    for lo, hi in ((0, 45), (45, 90), (90, 135), (135, 181)):
        groups.append((f'|θ| {lo}~{min(hi, 180)}°', keep & (a >= lo) & (a < hi)))
    rows = []
    for name, m in groups:
        m2 = m & near
        rows.append(dict(group=name, n=int(m.sum()), acc=float(ok[m].mean()) if m.any() else None,
                         n_near=int(m2.sum()), acc_near=float(ok[m2].mean()) if m2.any() else None,
                         gt_front=float((gc[m] > 0).mean()) if m.any() else None))
    return dict(rows=rows, n_cos0_dropped=int((~keep).sum()))


def prob_hist(kept, elem, thr):
    v = np.concatenate([_sig(out[:, :, elem])[_sig(out[:, :, E_ACT]) > thr] for _, out, _ in kept])
    cnt, edges = np.histogram(v, bins=np.linspace(0, 1, 11))
    return dict(edges=edges.round(2).tolist(), counts=cnt.tolist(), n=int(v.size), mean=float(v.mean()) if v.size else None,
                frac_04_06=float(((v >= 0.4) & (v <= 0.6)).mean()) if v.size else None)


def _rows(d):
    import pandas as pd
    fs = [f for f in os.listdir(d) if f.endswith('.csv')]
    return pd.concat([pd.read_csv(os.path.join(d, f)) for f in fs], ignore_index=True) if fs else None


def onscreen_stats(cfg, pred_dir, ref_dir, osa_now):
    import pandas as pd
    g_all, p_all = _rows(ref_dir), _rows(pred_dir)
    P = matched_pairs(pred_dir, ref_dir, fold=False)
    g, p = P[:, 2] > 0.5, P[:, 3] > 0.5
    tp = int((g & p).sum())
    prec, rec = tp / max(int(p.sum()), 1), tp / max(int(g.sum()), 1)
    zd = pred_dir + '_on0'
    if os.path.isdir(zd):
        shutil.rmtree(zd)
    os.makedirs(zd)
    for f in os.listdir(pred_dir):
        if f.endswith('.csv'):
            x = pd.read_csv(os.path.join(pred_dir, f))
            x['onscreen'] = 0
            x.to_csv(os.path.join(zd, f), index=False)
    osa0 = R.score(cfg, zd, ref_dir)['ext']['onscreen_acc']
    return dict(gt_on_ratio=float(g_all['onscreen'].mean()), pred_on_ratio=float(p_all['onscreen'].mean()) if p_all is not None and len(p_all) else None,
                n_pairs=int(len(P)), pair_gt_on_ratio=float(g.mean()) if len(P) else None, pair_pred_on_ratio=float(p.mean()) if len(P) else None,
                precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, EPS), acc_micro=float((g == p).mean()) if len(P) else None,
                OSA=osa_now, OSA_all0=osa0)


# ====================================================================== 1~4번 (N2)
def run_n2(cfg, log, combo=None):
    from .prerender import extract_labels
    path = os.path.join(out_dir(cfg), 'n2.json')
    if os.path.exists(path):
        log(f'[건너뜀] {path} (다시 하려면 지우세요)')
        return
    dev = R.device(); add_seld_path(cfg)
    t0 = time.time()
    model, ck = R.load_best(cfg, dev, log)
    col = np.degrees(model.vis.col_az.cpu().numpy())
    vmean = model.vis.vis_mean.detach().cpu()
    base_pp = R.postproc_params(cfg)
    thr = base_pp['act_thresh']
    refs = {s: extract_labels(cfg, s, log) for s in SPLITS}
    res = dict(meta=dict(run=cfg['run_name'], seed=cfg['seed'], ckpt=os.path.join(cfg['run_dir'], 'best_model.pth'), best_epoch=ck['epoch'] + 1,
                         postproc_json=base_pp, combos={k: dict(snap=v[0], onscreen_front=v[1]) for k, v in COMBOS.items()}),
               item1={}, item2={}, item3={}, fhist={}, item4={}, ohist={})
    ds, real = {}, {}
    for s in SPLITS:
        ds[s] = R.make_eval_ds(cfg, s, log, with_labels=False)
        real[s] = forward_all(model, VideoMode(ds[s], 'real', cfg['seed'], vmean), cfg, dev, True)
        res['item1'][s] = {}
        for k, (snap, front) in COMBOS.items():
            pp = dict(base_pp, snap_enable=snap, onscreen_front=front)
            d = tmp_dir(cfg, f'N2_{s}_real_{k}')
            sl = {} if snap else None
            decode_write(real[s], pp, col, d, cfg, sl)
            e = summarize(R.score(cfg, d, refs[s]))
            if snap:
                e['snap'] = snap_stats(sl, refs[s], real[s], thr)
            res['item1'][s][k] = e
            log(f'[1] {s} 조합 {k}: {e}')
    pick = lambda s: max(COMBOS, key=lambda k: -1 if res['item1'][s][k]['F_ext'] is None else res['item1'][s][k]['F_ext'])
    chosen = combo or pick('val')                      # combo 를 주면 그 조합으로 고정 (W1: 기존 postproc.json = 조합 a)
    res['chosen_val'], res['best_test'] = chosen, (pick('test') if 'test' in res['item1'] else None)
    pp_c = dict(base_pp, snap_enable=COMBOS[chosen][0], onscreen_front=COMBOS[chosen][1])
    res['meta']['postproc_chosen'] = pp_c
    log(f'val F_ext 최고 조합: {chosen} (test 최고: {res["best_test"]})')
    for s in SPLITS:
        res['item2'][s] = dict(real=res['item1'][s][chosen])
        for mode in ('zeros', 'shuffle'):
            kept = forward_all(model, VideoMode(ds[s], mode, cfg['seed'], vmean), cfg, dev, mode == 'shuffle')
            d = tmp_dir(cfg, f'N2_{s}_{mode}_{chosen}')
            decode_write(kept, pp_c, col, d, cfg)
            res['item2'][s][mode] = summarize(R.score(cfg, d, refs[s]))
            log(f'[2] {s} 영상 {mode}: {res["item2"][s][mode]}')
            del kept
        d_c = tmp_dir(cfg, f'N2_{s}_real_{chosen}')
        res['item3'][s] = fb_table(matched_pairs(d_c, refs[s], fold=True))
        res['fhist'][s] = prob_hist(real[s], E_FB, thr)
        res['item4'][s] = onscreen_stats(cfg, d_c, refs[s], res['item1'][s][chosen]['OSA'])
        res['ohist'][s] = prob_hist(real[s], E_ON, thr)
        log(f'[3,4] {s} 끝')
        real[s] = None
    res['meta']['seconds'] = round(time.time() - t0)
    R.atomic_json(path, res)
    log(f'저장: {path} ({time.time() - t0:.0f}초)')


# ====================================================================== 3번(N1-otf), 5번
def clip_vectors(pred_dir, ref_dir):
    """클립별 집계값 (2[ext, basic], 클립, 10, 13). for_dataset/build/eval_all.py 와 같은 계산."""
    from metrics import SELDMetrics
    from seld_label_utils import load_labels, organize_labels
    names = sorted(f[:-4] for f in os.listdir(ref_dir) if f.endswith('.csv'))
    arr = np.zeros((2, len(names), 10, 13))
    miss = 0
    for i, nm in enumerate(names):
        gt = organize_labels(load_labels(os.path.join(ref_dir, nm + '.csv'), convert_to_cartesian=False), CLIP_FRAMES)
        pp = os.path.join(pred_dir, nm + '.csv')
        if os.path.exists(pp):
            pr = organize_labels(load_labels(pp, convert_to_cartesian=False), CLIP_FRAMES)
        else:
            pr = organize_labels({}, CLIP_FRAMES); miss += 1
        for f in (0, 1):
            m = SELDMetrics(doa_threshold=20, dist_threshold=np.inf, reldist_threshold=1.0, req_onscreen=False, nb_classes=13,
                            average='macro', fold_azimuth=bool(f), exclude_absent_classes=True)
            m.update_seld_scores(pr, gt)
            arr[f, i] = np.stack([m._TP, m._FP, m._FP_spatial, m._FN, m._Nref, m._total_AngE, m._total_RelDistE, m._total_DistE,
                                  m._total_OnscreenCorrect, m._DE_TP])
    return names, arr, miss


def boot_metrics(S):
    """S: (2, 10, 13) 합 -> F_ext, F_basic, DOA_ext. for_dataset/build/analyze_results.py 의 metrics 와 같은 식."""
    out = {}
    for f, tag in ((0, 'ext'), (1, 'basic')):
        s = S[f]
        TP, FP, FPs, FN, Nref, de = s[0], s[1], s[2], s[3], s[4], s[9]
        F = TP / (EPS + TP + FPs + 0.5 * (FP + FN))
        F = np.where(Nref == 0, np.nan, F)
        out[f'F_{tag}'] = np.nanmean(F) if (~np.isnan(F)).any() else np.nan
        ang = np.where(de == 0, np.nan, s[5] / (de + EPS))
        out[f'DOA_{tag}'] = np.nanmean(ang) if (de > 0).any() else np.nan
    return out


def run_n1(cfg_pre, cfg_otf, log, out='n1'):
    """두 run 의 기존 예측 csv 비교. 키 이름은 기존 json 과 맞춰 pre(=A), otf(=B) 를 쓰고, 실제 프리셋은 meta.labels 에 둔다.
    W1: A=N1-otf, B=N1w (n1.json), A=N2, B=N2w (out='n1_N2' -> n1_N2.json)."""
    from .prerender import extract_labels
    path = os.path.join(out_dir(cfg_otf), f'{out}.json')
    if os.path.exists(path):
        log(f'[건너뜀] {path} (다시 하려면 지우세요)')
        return
    add_seld_path(cfg_otf)
    t0 = time.time()
    res = dict(meta=dict(pre=cfg_pre['run_name'], otf=cfg_otf['run_name'], labels=dict(pre=cfg_pre['preset'], otf=cfg_otf['preset']),
                         seed=cfg_otf['seed'], B=BOOT_B, boot_seed=BOOT_SEED,
                         postproc=dict(pre=R.postproc_params(cfg_pre), otf=R.postproc_params(cfg_otf))),
               item3={}, item5={})
    for s in SPLITS:
        ref = extract_labels(cfg_otf, s, log)
        pd_ = {k: os.path.join(c['run_dir'], f'pred_{s}') for k, c in (('pre', cfg_pre), ('otf', cfg_otf))}
        n_ref = len([f for f in os.listdir(ref) if f.endswith('.csv')])
        for k, d in pd_.items():
            n = len([f for f in os.listdir(d) if f.endswith('.csv')]) if os.path.isdir(d) else 0
            if n != n_ref:
                log(f'[주의] {d}: 예측 csv {n}개, 정답 {n_ref}개 (없는 클립은 빈 예측으로 채점)')
        res['item3'][s] = fb_table(matched_pairs(pd_['otf'], ref, fold=True))
        V = {}
        for k in ('pre', 'otf'):
            names, arr, miss = clip_vectors(pd_[k], ref)
            V[k] = arr
        point = {k: boot_metrics(V[k].sum(1)) for k in V}
        chk = {}
        for k, c in (('pre', cfg_pre), ('otf', cfg_otf)):
            sp = os.path.join(c['run_dir'], f'scores_{s}.json')
            if os.path.exists(sp):
                sc = json.load(open(sp, encoding='utf8'))['scores']
                chk[k] = max(abs(sc['ext']['F'] - point[k]['F_ext']), abs(sc['basic']['F'] - point[k]['F_basic']))
        rng = np.random.default_rng(BOOT_SEED)
        n = V['pre'].shape[1]
        diff = {m: np.zeros(BOOT_B) for m in ('F_ext', 'F_basic')}
        for b in range(BOOT_B):
            idx = rng.integers(0, n, n)
            a, o = boot_metrics(V['pre'][:, idx].sum(1)), boot_metrics(V['otf'][:, idx].sum(1))
            for m in diff:
                diff[m][b] = o[m] - a[m]
        e = dict(n_clips=n, check_vs_scores_json=chk)
        for m in diff:
            x = diff[m][~np.isnan(diff[m])]
            e[m] = dict(pre=float(point['pre'][m]), otf=float(point['otf'][m]), diff=float(point['otf'][m] - point['pre'][m]),
                        ci95=[float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))], frac_gt0=float((x > 0).mean()))
        res['item5'][s] = e
        log(f'[5] {s}: {e}')
    R.atomic_json(path, res)
    log(f'저장: {path} ({time.time() - t0:.0f}초)')


# ====================================================================== 표 (tables.md)
def _p(v, k=2):
    return '–' if v is None else f'{100 * v:.{k}f}'


def _n(v, k=2):
    return '–' if v is None else f'{v:.{k}f}'


def write_tables(cfg, log=print):
    od = out_dir(cfg)
    n2 = json.load(open(os.path.join(od, 'n2.json'), encoding='utf8')) if os.path.exists(os.path.join(od, 'n2.json')) else None
    n1 = json.load(open(os.path.join(od, 'n1.json'), encoding='utf8')) if os.path.exists(os.path.join(od, 'n1.json')) else None
    pairs = [json.load(open(os.path.join(od, f), encoding='utf8')) for f in sorted(os.listdir(od)) if f.startswith('n1') and f.endswith('.json')] if os.path.isdir(od) else []
    N = cfg['preset']                                                    # 진단한 2단계 프리셋 (N2 또는 N2w)
    lab = lambda j, k, d: j['meta'].get('labels', {}).get(k, d)          # DIAG_01 json 에는 labels 가 없다
    A1 = lab(n1, 'otf', 'N1-otf') if n1 else 'N1-otf'
    L = [f'# {DIAG_ID} 결과 표 (자동 생성)', '']
    if n2:
        m = n2['meta']; ppj = m['postproc_json']
        L += [f"- {N} 체크포인트: `{m['run']}/best_model.pth` (에폭 {m['best_epoch']}), 시드 {m['seed']}",
              f"- 후처리 기본값(postproc.json): τ={ppj['onscreen_tau']}, 열 점수 기준={ppj['snap_score']}, 창 ±{ppj['snap_window_deg']}°, "
              f"활성 {ppj['act_thresh']}, 통합 {ppj['merge_deg']}° (postproc.json 의 보정 켜기={ppj['snap_enable']})",
              f"- val F_ext 최고 조합: **{n2['chosen_val']}**, test F_ext 최고 조합: {n2['best_test']}", '',
              f'## 1. {N} 후처리 4조합', '']
        for s in SPLITS:
            L += [f'**{s}**', '', '| 조합 | 위치 보정 | on→앞 | F_ext | F_basic | DOAE_ext | RDE | OSA | 보정이 바꾼 비율 | 바뀐 것 중 가까워짐 (정답 있는 것 중) |',
                  '|---|---|---|---|---|---|---|---|---|---|']
            for k, (sn, fr) in COMBOS.items():
                e = n2['item1'][s][k]; sp = e.get('snap')
                sc = f"{_p(sp['changed_ratio'])}% ({sp['n_changed']}/{sp['n_active']})" if sp else '–'
                cl = f"{_p(sp['closer_ratio'])}% ({_p(sp['closer_ratio_with_gt'])}%)" if sp else '–'
                L.append(f"| {k} | {'켬' if sn else '끔'} | {'켬' if fr else '끔'} | {_p(e['F_ext'])} | {_p(e['F_basic'])} | {_n(e['DOAE_ext'])} | "
                         f"{_n(e['RDE'], 4)} | {_p(e['OSA'])} | {sc} | {cl} |")
            L.append('')
        L += [f"## 2. {N} 영상 원래 / 0 / 섞기 (후처리 조합 {n2['chosen_val']})", '',
              '| split | 영상 | F_ext | F_basic | DOAE_ext | RDE | OSA |', '|---|---|---|---|---|---|---|']
        for s in SPLITS:
            for mode, nm in (('real', '원래'), ('zeros', '0'), ('shuffle', '섞기')):
                e = n2['item2'][s][mode]
                L.append(f"| {s} | {nm} | {_p(e['F_ext'])} | {_p(e['F_basic'])} | {_n(e['DOAE_ext'])} | {_n(e['RDE'], 4)} | {_p(e['OSA'])} |")
        L.append('')
    if n1 or n2:
      L += ['## 3. 앞/뒤 정확도 (접은 방위각 매칭 쌍; 괄호 = 접은 오차 ≤20° 쌍)', '',
            f'기준선: {FB_BASELINE:.3f} (이보다 낮으면 앞/뒤 둘 다 내는 쪽이 점수상 유리). {A1} 는 앞/뒤를 학습하지 않아 항상 앞으로 낸다.', '']
      for s in SPLITS:
        L += [f'**{s}**', '', f'| 그룹 | {A1} 정확도 | {N} 정확도 | 쌍 수 ({N}) | 정답 앞 비율 ({N} 쌍) |', '|---|---|---|---|---|']
        a = n1['item3'][s]['rows'] if n1 else None
        b = n2['item3'][s]['rows'] if n2 else None
        for i in range(len((a or b))):
            ra, rb = (a[i] if a else None), (b[i] if b else None)
            f = lambda r: '–' if r is None else f"{_p(r['acc'], 1)} ({_p(r['acc_near'], 1)})"
            L.append(f"| {(ra or rb)['group']} | {f(ra)} | {f(rb)} | {rb['n'] if rb else '–'} ({rb['n_near'] if rb else '–'}) | {_p(rb['gt_front'], 1) if rb else '–'} |")
        L.append('')
    if n2:
        L += [f'{N} 앞/뒤 확률 f 분포 (활성 칸, 10구간 개수)', '', '| split | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.4~0.6 비율 |',
              '|---|' + '---|' * 11]
        for s in SPLITS:
            h = n2['fhist'][s]
            L.append(f"| {s} | " + ' | '.join(str(c) for c in h['counts']) + f" | {_p(h['frac_04_06'], 1)}% |")
        L += ['', f"## 4. {N} onscreen 헤드 (조합 {n2['chosen_val']}; 정밀도/재현율은 ext 매칭 쌍 기준)", '',
              '| split | 정답 on 비율 (전체 행) | 예측 on 비율 (전체 행) | 정밀도 | 재현율 | F1 | OSA | 전부 0일 때 OSA |', '|---|---|---|---|---|---|---|---|']
        for s in SPLITS:
            e = n2['item4'][s]
            L.append(f"| {s} | {_p(e['gt_on_ratio'], 1)} | {_p(e['pred_on_ratio'], 1)} | {_p(e['precision'], 1)} | {_p(e['recall'], 1)} | "
                     f"{_p(e['f1'], 1)} | {_p(e['OSA'], 1)} | {_p(e['OSA_all0'], 1)} |")
        L += ['', 'onscreen 확률 o 분포 (활성 칸, 10구간 개수)', '', '| split | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |', '|---|' + '---|' * 10]
        for s in SPLITS:
            L.append(f"| {s} | " + ' | '.join(str(c) for c in n2['ohist'][s]['counts']) + ' |')
        L.append('')
    for k5, P in enumerate(pairs):
        a5, b5 = lab(P, 'pre', 'N1-pre'), lab(P, 'otf', 'N1-otf')
        L += [f"## 5{'-' + str(k5 + 1) if len(pairs) > 1 else ''}. {a5} vs {b5} 부트스트랩 (클립 단위 {P['meta']['B']}번, 시드 {P['meta']['boot_seed']})", '',
              f"`{P['meta']['pre']}` vs `{P['meta']['otf']}`, 기존 예측 csv (runs/*/pred_val, pred_test, best 체크포인트)", '',
              f'| split | 지표 | {a5} | {b5} | {b5} − {a5} | 95% 구간 | {b5}>{a5} 비율 |', '|---|---|---|---|---|---|---|']
        for s in SPLITS:
            e = P['item5'][s]
            for m in ('F_ext', 'F_basic'):
                x = e[m]
                L.append(f"| {s} | {m} | {_p(x['pre'])} | {_p(x['otf'])} | {_p(x['diff'])} | [{_p(x['ci95'][0])}, {_p(x['ci95'][1])}] | {_p(x['frac_gt0'], 1)}% |")
        L.append('')
    p = os.path.join(od, 'tables.md')
    os.makedirs(od, exist_ok=True)
    open(p, 'w', encoding='utf8').write('\n'.join(L))
    log(f'표 저장: {p}')
    return p
