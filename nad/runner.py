"""
runner.py -- 학습, 평가, 예측, 채점, 확인 단계.

이어 하기: 에폭(= steps_per_epoch 번 갱신)마다 {run_dir}/last.pth 에 모델, 옵티마이저, 스케줄러, 스케일러, 에폭, 갱신 횟수,
최고 점수를 저장하고 train_log.csv 에 한 줄 더한다. 같은 명령을 다시 실행하면 마지막 에폭 다음부터 이어 한다.
화면 출력은 {run_dir}/train_out.log 에도 쌓인다.
"""
import os
import csv
import json
import math
import time
import shutil
import random
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import add_seld_path, CLIP_FRAMES
from .model import NADModel
from .loss import NADLoss
from .postproc import decode_clip, write_clip

PP_KEYS = ('act_thresh', 'onscreen_tau', 'snap_enable', 'snap_score', 'snap_window_deg', 'merge_deg')
VISUAL_PREFIX = ('vis.', 'rot.', 'lca.')


class Logger:
    def __init__(self, path=None):
        self.path = path

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'a', encoding='utf8') as f:
                f.write(time.strftime('%m-%d %H:%M:%S ') + s + '\n')


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def amp_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def modality(cfg):
    return 'audio_visual' if cfg['stage'] == 2 else 'audio'


def postproc_params(cfg, run_dir=None):
    pp = {k: cfg[k] for k in PP_KEYS}
    p = os.path.join(run_dir or cfg['run_dir'], 'postproc.json')
    if os.path.exists(p):
        pp.update(json.load(open(p, encoding='utf8'))['params'])
    if cfg['stage'] == 1:
        pp['snap_enable'] = False
    return pp


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)
    os.replace(path + '.tmp', path)


# ====================================================================== 데이터 준비
def make_clap(cfg, keys):
    from .semantic import ClapLookup
    return ClapLookup(cfg, keys)


def make_video(cfg, keys, log):
    if cfg['stage'] != 2:
        return None
    from .data import VideoStore
    return VideoStore(cfg, keys, log)


def train_keys(cfg):
    return [{0: 'train_s0', 1: 'train_s1'}[s] for s in cfg['train_stages']]


def make_train_ds(cfg, log):
    from .data import PreTrainDataset, OTFTrainDataset
    n = cfg['steps_per_epoch'] * cfg['batch_size']
    if cfg['data_source'] == 'onthefly':
        from .onthefly import RecordingBank
        return OTFTrainDataset(cfg, n, RecordingBank(cfg, log), make_clap(cfg, ['rec']))
    keys = train_keys(cfg)
    return PreTrainDataset(cfg, n, keys, make_clap(cfg, keys), make_video(cfg, keys, log))


def make_eval_ds(cfg, key, log, with_labels=True):
    from .data import EvalDataset
    return EvalDataset(cfg, key, make_clap(cfg, [key]), make_video(cfg, [key], log), with_labels)


def loader(ds, cfg, shuffle=False, drop_last=False, bs=None):
    return DataLoader(ds, batch_size=bs or cfg['batch_size'], shuffle=shuffle, drop_last=drop_last, num_workers=cfg['num_workers'],
                      pin_memory=torch.cuda.is_available(), persistent_workers=False)


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


# ====================================================================== 모델 준비
def build_model(cfg, dev, log):
    model = NADModel(cfg).to(dev)
    if cfg['stage'] == 2:
        from .visual import load_meta
        model.set_visual_meta(load_meta(cfg))
    return model


def load_init(model, cfg, dev, log):
    """2단계: 1단계 최고 모델에서 (A),(B),CMC,(E-2),(F)와 특징 통계를 가져온다(영상 모듈은 새로 시작)."""
    if not cfg['init_from']:
        if cfg['stage'] == 2:
            log('[경고] init_from 이 비어 있습니다. 2단계를 처음부터 학습합니다(1단계 가중치 없음).')
        return False
    p = os.path.join(cfg['out_root'], cfg['init_from'], 'best_model.pth')
    sd = torch.load(p, map_location=dev, weights_only=False)['model']
    sd = {k: v for k, v in sd.items() if not k.startswith(VISUAL_PREFIX)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.feat.has_stats = True
    log(f'1단계 가중치 불러옴: {p}  (새로 시작하는 영상 모듈 파라미터 {len(missing)}개)')
    return True


def fit_feature_stats(model, ds, cfg, dev, log):
    """학습 데이터의 처음 몇 배치로 특징 정규화 통계(채널, mel별)를 구한다."""
    path = os.path.join(cfg['run_dir'], 'feat_stats.pt')
    if os.path.exists(path):
        model.feat.load_stats(torch.load(path, weights_only=False)); log(f'특징 통계 불러옴: {path}'); return
    ds.set_epoch(10 ** 6)
    it = iter(loader(ds, cfg))
    def batches():
        for _ in range(cfg['feat_stats_batches']):
            yield next(it)['audio']
    t0 = time.time()
    model.feat.fit_stats(batches())
    torch.save(model.feat.stats_dict(), path)
    log(f'특징 통계 {cfg["feat_stats_batches"]}배치로 구함 ({time.time() - t0:.0f}초) -> {path}')


def lr_lambda(cfg):
    W, T = cfg['warmup_steps'], cfg['total_steps']
    def f(step):
        if step < W:
            return (step + 1) / W
        p = min(1.0, (step - W) / max(1, T - W))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))
    return f


def make_optimizer(model, cfg):
    sp = list(model.spatial.parameters())
    sp_ids = {id(p) for p in sp}
    rest = [p for p in model.parameters() if id(p) not in sp_ids]
    lr_sp = cfg['finetune_lr'] if cfg['stage'] == 2 else cfg['lr']
    opt = torch.optim.AdamW([dict(params=sp, lr=lr_sp), dict(params=rest, lr=cfg['lr'])], weight_decay=cfg['weight_decay'])
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    return opt, sch


def set_spatial_frozen(model, frozen):
    for p in model.spatial.parameters():
        p.requires_grad = not frozen


# ====================================================================== 예측과 채점
@torch.no_grad()
def predict(model, ds, cfg, dev, out_dir, pp, loss_fn=None, keep_outputs=False):
    """ds 전체를 예측해 out_dir 에 csv 를 쓴다. 반환: (평균 손실 또는 None, [원출력] 또는 None)"""
    add_seld_path(cfg)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    use_video = cfg['stage'] == 2
    col = np.degrees(model.vis.col_az.cpu().numpy()) if use_video else None
    tot, nb, kept = 0.0, 0, []
    for b in loader(ds, cfg, bs=cfg['batch_size']):
        b = to_dev(b, dev)
        with torch.autocast('cuda', dtype=amp_dtype(), enabled=cfg['amp'] and dev == 'cuda'):
            pred = model(b['audio'], b['clap'], b.get('video') if use_video else None)
        if loss_fn is not None and 'labels' in b:
            tot += loss_fn(pred, b['labels'], b.get('dpsi'))[0].item(); nb += 1
        out = pred['out'].float().cpu().numpy()
        vid = b['video'].float().cpu().numpy() if use_video else None
        for i, n in enumerate(b['name']):
            if keep_outputs:
                kept.append((n, out[i], None if vid is None else vid[i].astype(np.float16)))
            d = decode_clip(out[i], pp, vid[i] if use_video else None, col, fb_trained=cfg['loss_fb'], on_trained=cfg['loss_onscreen'])
            write_clip(d, out_dir, n)
    return (tot / nb if nb else None), (kept if keep_outputs else None)


def score(cfg, pred_dir, ref_dir):
    add_seld_path(cfg)
    from score import score_dirs
    return score_dirs(pred_dir, ref_dir, CLIP_FRAMES, modality(cfg), None, exclude_absent_classes=True)


def one_line(res):
    from score import one_line as ol            # score() 가 먼저 add_seld_path 를 부른다
    return ol(res)


def metric(res, name):
    v = res['ext' if name == 'F_ext' else 'basic']['F']
    return float('-inf') if v is None else v


# ====================================================================== train
def stage_train(cfg):
    from .prerender import extract_labels
    dev = device()
    os.makedirs(cfg['run_dir'], exist_ok=True)
    log = Logger(os.path.join(cfg['run_dir'], 'train_out.log'))
    set_seed(cfg['seed'])
    import yaml
    yaml.safe_dump(dict(cfg), open(os.path.join(cfg['run_dir'], 'config_used.yaml'), 'w', encoding='utf8'), allow_unicode=True, sort_keys=False)
    log(f'== 학습: {cfg["run_name"]} (프리셋 {cfg["preset"]}, 데이터 {cfg["data_source"]}, 단계 {cfg["stage"]}, 장치 {dev})')

    train_ds = make_train_ds(cfg, log)
    val_ds = make_eval_ds(cfg, 'val', log)
    val_ref = extract_labels(cfg, 'val', log)
    model = build_model(cfg, dev, log)
    inited = load_init(model, cfg, dev, log)
    if not inited:
        fit_feature_stats(model, train_ds, cfg, dev, log)
    else:
        torch.save(model.feat.stats_dict(), os.path.join(cfg['run_dir'], 'feat_stats.pt'))
    frozen = cfg['stage'] == 2 and cfg['freeze_spatial']
    set_spatial_frozen(model, frozen)
    loss_fn = NADLoss(cfg)
    opt, sch = make_optimizer(model, cfg)
    use_scaler = cfg['amp'] and dev == 'cuda' and amp_dtype() == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    n_par = sum(p.numel() for p in model.parameters()); n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'파라미터 {n_par / 1e6:.1f}M (학습 {n_tr / 1e6:.1f}M), train 항목/에폭 {len(train_ds)}, val {len(val_ds)}')

    last_p, best_p = os.path.join(cfg['run_dir'], 'last.pth'), os.path.join(cfg['run_dir'], 'best_model.pth')
    log_p = os.path.join(cfg['run_dir'], 'train_log.csv')
    start_epoch, step, best, best_epoch = 0, 0, float('-inf'), -1
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location=dev, weights_only=False)
        if cfg['stage'] == 2 and (ck.get('init_from') or '') != (cfg['init_from'] or ''):
            raise SystemExit(f'[중단] {last_p} 는 init_from={ck.get("init_from") or "(없음)"} 으로 학습한 체크포인트인데, '
                             f'지금 init_from={cfg["init_from"] or "(없음)"} 입니다. 이 실행 폴더를 지우거나 이름을 바꾼 뒤 다시 실행하세요.')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sch.load_state_dict(ck['sch'])
        if use_scaler and ck.get('scaler'):
            scaler.load_state_dict(ck['scaler'])
        start_epoch, step, best, best_epoch = ck['epoch'] + 1, ck['step'], ck['best'], ck['best_epoch']
        if cfg['stage'] == 2 and cfg['finetune_after_step'] and step >= cfg['finetune_after_step']:
            frozen = False; set_spatial_frozen(model, False)
        log(f'이어 하기: 에폭 {start_epoch + 1}, 갱신 {step}번부터 (최고 {cfg["select_metric"]}={best:.4f}, 에폭 {best_epoch + 1})')
    elif os.path.exists(log_p):
        os.remove(log_p)

    n_epochs = math.ceil(cfg['total_steps'] / cfg['steps_per_epoch'])
    tmp_pred = os.path.join(cfg['local_root'], 'tmp', cfg['run_name'], 'pred_val')
    pp = postproc_params(cfg)
    cols = ['epoch', 'step', 'lr', 'train_loss', 'act', 'sin', 'dist', 'fb', 'on', 'rot', 'val_loss',
            'F_ext', 'F_basic', 'DOA_ext', 'DOA_basic', 'RDE_ext', 'OSA_ext', 'seconds']
    use_video = cfg['stage'] == 2
    for epoch in range(start_epoch, n_epochs):
        if step >= cfg['total_steps']:
            break
        t0 = time.time()
        model.train()
        if frozen:
            model.spatial.eval()
        train_ds.set_epoch(epoch)
        sums, nb = {}, 0
        for b in loader(train_ds, cfg, drop_last=True):
            if cfg['stage'] == 2 and frozen and cfg['finetune_after_step'] and step >= cfg['finetune_after_step']:
                frozen = False; set_spatial_frozen(model, False); model.spatial.train()
                log(f'갱신 {step}: (A) 인코더 고정 해제 (lr {cfg["finetune_lr"]})')
            b = to_dev(b, dev)
            with torch.autocast('cuda', dtype=amp_dtype(), enabled=cfg['amp'] and dev == 'cuda'):
                pred = model(b['audio'], b['clap'], b.get('video') if use_video else None)
            loss, parts = loss_fn(pred, b['labels'], b.get('dpsi'))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['grad_clip'])
            scaler.step(opt); scaler.update(); sch.step()
            step += 1; nb += 1
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            if nb % 100 == 0:
                log(f'  에폭 {epoch + 1} 갱신 {nb}/{cfg["steps_per_epoch"]} 손실 {sums["total"] / nb:.4f} ({time.time() - t0:.0f}초)')
            if step >= cfg['total_steps']:
                break
        val_loss, _ = predict(model, val_ds, cfg, dev, tmp_pred, pp, loss_fn)
        res = score(cfg, tmp_pred, val_ref)
        cur = metric(res, cfg['select_metric'])
        e, bb = res['ext'], res['basic']
        row = dict(epoch=epoch + 1, step=step, lr=sch.get_last_lr()[-1], train_loss=sums.get('total', 0) / max(1, nb),
                   **{k: (sums[k] / nb if k in sums else None) for k in ('act', 'sin', 'dist', 'fb', 'on', 'rot')},
                   val_loss=val_loss, F_ext=e['F'], F_basic=bb['F'], DOA_ext=e['DOA_err'], DOA_basic=bb['DOA_err'],
                   RDE_ext=e['rel_dist_err'], OSA_ext=e['onscreen_acc'], seconds=time.time() - t0)
        new = not os.path.exists(log_p)
        with open(log_p, 'a', newline='', encoding='utf8') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerow(row)
        pct = lambda v: 'NA' if v is None else f'{100 * v:.2f}'
        log(f'에폭 {epoch + 1}/{n_epochs} (갱신 {step}) | 학습손실 {row["train_loss"]:.4f} | val손실 {val_loss:.4f} | '
            f'F_ext {pct(e["F"])} F_basic {pct(bb["F"])} | DOA_ext {e["DOA_err"]} | {row["seconds"]:.0f}초')
        if cur >= best:
            best, best_epoch = cur, epoch
            torch.save(dict(model=model.state_dict(), epoch=epoch, step=step, best=best, val_scores=res), best_p + '.tmp')
            os.replace(best_p + '.tmp', best_p)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), scaler=scaler.state_dict() if use_scaler else None,
                        epoch=epoch, step=step, best=best, best_epoch=best_epoch, init_from=cfg['init_from'] or ''), last_p + '.tmp')
        os.replace(last_p + '.tmp', last_p)
    atomic_json(os.path.join(cfg['run_dir'], 'train_summary.json'),
                dict(run=cfg['run_name'], best_epoch=best_epoch + 1, best_value=best, select_metric=cfg['select_metric'], total_steps=step))
    log(f'== 학습 끝. 최고 {cfg["select_metric"]}={best:.4f} (에폭 {best_epoch + 1})')


# ====================================================================== eval / tune / predict / score
def load_best(cfg, dev, log):
    model = build_model(cfg, dev, log)
    ck = torch.load(os.path.join(cfg['run_dir'], 'best_model.pth'), map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    return model, ck


def stage_eval(cfg, split='val'):
    """최고 모델로 val 을 예측, 채점한다 (후처리 값은 postproc.json 이 있으면 그것)."""
    from .prerender import extract_labels
    dev = device(); log = Logger(os.path.join(cfg['run_dir'], 'train_out.log'))
    model, ck = load_best(cfg, dev, log)
    pp = postproc_params(cfg)
    out = os.path.join(cfg['run_dir'], 'pred_val')
    predict(model, make_eval_ds(cfg, 'val', log), cfg, dev, out, pp)
    res = score(cfg, out, extract_labels(cfg, 'val', log))
    atomic_json(os.path.join(cfg['run_dir'], 'scores_val.json'), dict(run=cfg['run_name'], best_epoch=ck['epoch'] + 1, postproc=pp, scores=res))
    log(f'[val] {one_line(res)}  (후처리 {pp})')
    return res


def stage_tune_post(cfg):
    """val 로 후처리 값(τ, 열 점수 기준, ±창, 보정 켜기/끄기)을 고른다. 모델 출력은 한 번만 계산한다."""
    import itertools
    from .prerender import extract_labels
    dev = device(); log = Logger(os.path.join(cfg['run_dir'], 'train_out.log'))
    model, _ = load_best(cfg, dev, log)
    ref = extract_labels(cfg, 'val', log)
    base = {k: cfg[k] for k in PP_KEYS}
    tmp = os.path.join(cfg['local_root'], 'tmp', cfg['run_name'], 'tune')
    _, kept = predict(model, make_eval_ds(cfg, 'val', log), cfg, dev, tmp, base, keep_outputs=True)
    col = np.degrees(model.vis.col_az.cpu().numpy())
    grid = [dict(base, snap_enable=False, onscreen_tau=t) for t in (0.3, 0.5, 0.7)]
    if cfg['stage'] == 2:
        grid += [dict(base, snap_enable=True, onscreen_tau=t, snap_score=s, snap_window_deg=w)
                 for t, s, w in itertools.product((0.3, 0.5, 0.7), (0.05, 0.1, 0.2, 0.3), (5.0, 10.0, 15.0))]
    rows, best = [], None
    for g in grid:
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        for n, out, vid in kept:
            write_clip(decode_clip(out, g, None if vid is None else vid.astype(np.float32), col, cfg['loss_fb'], cfg['loss_onscreen']), tmp, n)
        r = score(cfg, tmp, ref)
        v = metric(r, 'F_ext')
        rows.append(dict(g, F_ext=r['ext']['F'], F_basic=r['basic']['F'], DOA_ext=r['ext']['DOA_err']))
        if best is None or v > best[0]:
            best = (v, g)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(cfg['run_dir'], 'postproc_grid.csv'), index=False)
    atomic_json(os.path.join(cfg['run_dir'], 'postproc.json'), dict(params={k: best[1][k] for k in PP_KEYS}, val_F_ext=best[0]))
    log(f'후처리 고름 (val F_ext {100 * best[0]:.2f}): {best[1]}  -> postproc.json, 전체 표 postproc_grid.csv')


def stage_predict(cfg):
    dev = device(); log = Logger(os.path.join(cfg['run_dir'], 'train_out.log'))
    model, _ = load_best(cfg, dev, log)
    pp = postproc_params(cfg)
    out = os.path.join(cfg['run_dir'], 'pred_test')
    ds = make_eval_ds(cfg, 'test', log, with_labels=False)
    predict(model, ds, cfg, dev, out, pp)
    z = os.path.join(cfg['run_dir'], f'submission_{cfg["run_name"]}.zip')
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(out)):
            zf.write(os.path.join(out, f), f)
    log(f'test 예측 {len(ds)}개 -> {out}, 제출 zip {z}')


def stage_score_test(cfg):
    """최종 보고용: 비공개 test 정답(private_test.zip)으로 채점한다. 모델/후처리 선택에는 쓰지 않는다."""
    from .prerender import extract_labels
    log = Logger(os.path.join(cfg['run_dir'], 'train_out.log'))
    res = score(cfg, os.path.join(cfg['run_dir'], 'pred_test'), extract_labels(cfg, 'test', log))
    atomic_json(os.path.join(cfg['run_dir'], 'scores_test.json'), dict(run=cfg['run_name'], postproc=postproc_params(cfg), scores=res))
    log(f'[test] {one_line(res)}')
    return res


# ====================================================================== 확인: 출력 형식 -> csv -> 채점
def oracle_check(cfg, n_clips=None, log=print):
    """val 정답을 새 출력 헤드 형식(활성 함수 전 값)으로 바꾸고, 후처리 -> csv -> 베이스라인 채점이 정상인지 본다.
    정답을 그대로 넣었으므로 F 가 1 에 가까워야 한다(같은 클래스가 20° 안에 겹치면 합쳐져서 조금 낮다)."""
    import pandas as pd
    from .prerender import extract_labels
    from .data import events_to_adpit
    from .model import E_ACT, E_SIN, E_FB, E_DIST, E_ON
    add_seld_path(cfg)
    ref = extract_labels(cfg, 'val', log)
    files = sorted(os.listdir(ref))[:n_clips] if n_clips else sorted(os.listdir(ref))
    tmp_ref = os.path.join(cfg['local_root'], 'tmp', 'oracle_ref'); tmp_pred = os.path.join(cfg['local_root'], 'tmp', 'oracle_pred')
    for d in (tmp_ref, tmp_pred):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    pp = {k: cfg[k] for k in PP_KEYS}; pp['snap_enable'] = False
    for f in files:
        shutil.copy(os.path.join(ref, f), tmp_ref)
        ev = pd.read_csv(os.path.join(ref, f))[['frame', 'class', 'source', 'azimuth', 'distance', 'onscreen']].values.astype(np.int32)
        y = events_to_adpit(ev).numpy()                                  # (T, 6, 5, C): act, x, y, dist, on
        out = np.full((CLIP_FRAMES, 3, 5, y.shape[-1]), -10.0, np.float32)
        for g, tracks in ((0, [0, 0, 0]), (1, [1, 1, 2]), (3, [3, 4, 5])):
            for k, src in enumerate(tracks):
                m = y[:, src, 0] > 0.5
                if g == 0:
                    m = m & (y[:, 1, 0] < 0.5) & (y[:, 3, 0] < 0.5)
                tt, cc = np.nonzero(m)
                out[tt, k, E_ACT, cc] = 10.0
                out[tt, k, E_SIN, cc] = np.arctanh(np.clip(y[tt, src, 2, cc], -0.999999, 0.999999))
                out[tt, k, E_FB, cc] = np.where(y[tt, src, 1, cc] > 0, 10.0, -10.0)
                out[tt, k, E_DIST, cc] = np.log(np.maximum(y[tt, src, 3, cc], 0.05))
                out[tt, k, E_ON, cc] = np.where(y[tt, src, 4, cc] > 0.5, 10.0, -10.0)
        write_clip(decode_clip(out, pp, fb_trained=True, on_trained=True), tmp_pred, f[:-4])
    from score import score_dirs, one_line as ol
    res = score_dirs(tmp_pred, tmp_ref, CLIP_FRAMES, 'audio_visual', None, exclude_absent_classes=True)
    log(f'[정답 -> 출력 형식 -> csv -> 채점] 클립 {len(files)}개: {ol(res)}')
    return res
