"""
prerender.py -- 미리 렌더링한 데이터(main20 train/val/test, S1)를 학습에 쓰기 좋은 캐시로 만든다.

 - 오디오: zip 안의 wav 만 읽어 int16 메모리맵 한 파일로 (코랩 로컬 디스크, 세션마다 다시 만든다)
     {local_root}/audio/{key}.npy  (N, 2, 480000) int16,  {key}_names.json
 - 라벨, 궤적: 클립마다 (사건 배열, 프레임별 ψ, 거울 여부, 원본 녹음, 시작 프레임)  -> 드라이브 {cache_root}/meta/{key}.pkl (작음)
 - val 라벨 csv: 채점용으로 로컬 {local_root}/data/val/labels/ 에 푼다
키: train_s0 (main20 train 2999), train_s1 (S1 5912), val (123), test (526, 라벨 없음)
for_dataset 의 zip 은 읽기만 한다.
"""
import os
import io
import re
import glob
import json
import time
import pickle
import zipfile
import numpy as np
import pandas as pd
import soundfile as sf

from .config import fd_paths, CLIP_FRAMES

N_SAMP = 480000
S0_RE = re.compile(r'^train_\d+$')


def zip_sources(cfg, key):
    """키의 (zip 경로 목록, 이름 접두 폴더)."""
    p = fd_paths(cfg)
    if key == 'train_s0':
        return [os.path.join(p['main20'], 'train.zip')], 'train'
    if key == 'train_s1':
        return sorted(glob.glob(os.path.join(p['s1'], 's1_main20_train_p*.zip'))), 'train'
    if key == 'val':
        return [os.path.join(p['main20'], 'val.zip')], 'val'
    if key == 'test':
        return [os.path.join(p['main20'], 'test_inputs.zip')], 'test'
    raise ValueError(key)


def _name_ok(key, name):
    if key == 'train_s0':
        return bool(S0_RE.match(name))
    if key == 'train_s1':
        return name.startswith('train_s1_')
    return True


def list_members(cfg, key):
    """{클립 이름: (zip 경로, {audio, video, labels, traj} 멤버 이름)}  (이름 순)"""
    zips, top = zip_sources(cfg, key)
    out = {}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            names = set(zf.namelist())
            for m in names:
                if m.startswith(f'{top}/audio/') and m.endswith('.wav'):
                    n = os.path.splitext(os.path.basename(m))[0]
                    if not _name_ok(key, n):
                        continue
                    d = dict(audio=m, video=f'{top}/video/{n}.mp4', labels=f'{top}/labels/{n}.csv', traj=f'{top}/meta/{n}_traj.csv')
                    out[n] = (z, {k: v for k, v in d.items() if v in names})
    return dict(sorted(out.items()))


def audio_path(cfg, key):
    return os.path.join(cfg['local_root'], 'audio', f'{key}.npy')


def size_report(cfg, keys, log=print):
    tot = 0
    for k in keys:
        n = len(list_members(cfg, k))
        gb = n * 2 * N_SAMP * 2 / 2 ** 30
        tot += gb
        log(f'  {k}: 클립 {n}개 -> 오디오 캐시 {gb:.1f}GB (코랩 로컬)')
    log(f'  합계 {tot:.1f}GB (로컬 디스크). 라벨 캐시는 드라이브에 수십 MB.')
    return tot


# ====================================================================== 오디오 캐시
def build_audio(cfg, key, log=print):
    path = audio_path(cfg, key)
    done = path + '.done'
    if os.path.exists(done):
        log(f'[건너뜀] 오디오 캐시 있음: {path}')
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mem = list_members(cfg, key)
    names = list(mem)
    prog_p = path + '.progress.json'
    if os.path.exists(path) and os.path.exists(prog_p):
        arr = np.load(path, mmap_mode='r+')
        start = json.load(open(prog_p))['done']
    else:
        arr = np.lib.format.open_memmap(path, mode='w+', dtype=np.int16, shape=(len(names), 2, N_SAMP))
        start = 0
    json.dump(names, open(path.replace('.npy', '_names.json'), 'w'))
    t0 = time.time()
    by_zip = {}
    for i, n in enumerate(names):
        by_zip.setdefault(mem[n][0], []).append(i)
    for z, idx in by_zip.items():
        idx = [i for i in idx if i >= start]
        if not idx:
            continue
        with zipfile.ZipFile(z) as zf:
            for c, i in enumerate(idx):
                x, sr = sf.read(io.BytesIO(zf.read(mem[names[i]][1]['audio'])), dtype='int16', always_2d=True)
                assert sr == 24000 and x.shape == (N_SAMP, 2), (names[i], x.shape)
                arr[i] = x.T
                if (i + 1) % 250 == 0:
                    arr.flush(); json.dump(dict(done=i + 1), open(prog_p, 'w'))
                    log(f'  {key}: {i + 1}/{len(names)} ({time.time() - t0:.0f}초)')
    arr.flush(); del arr
    open(done, 'w').write(str(time.time()))
    if os.path.exists(prog_p):
        os.remove(prog_p)
    log(f'  {key}: 오디오 캐시 완료 {len(names)}개 ({time.time() - t0:.0f}초)')


def load_audio(cfg, key):
    path = audio_path(cfg, key)
    if not os.path.exists(path + '.done'):
        raise FileNotFoundError(f'오디오 캐시가 없습니다: {path}  (prepare_pre 를 먼저 실행)')
    return json.load(open(path.replace('.npy', '_names.json'))), np.load(path, mmap_mode='r')


# ====================================================================== 라벨/궤적 캐시
def meta_path(cfg, key):
    return os.path.join(cfg['cache_root'], 'meta', f'{key}.pkl')


def _manifest(cfg, key):
    """원본 녹음, 시작 프레임, 거울 여부 (S0 은 main20 manifest.csv, S1 은 s1 manifest)."""
    p = fd_paths(cfg)
    if key == 'train_s0':
        m = pd.read_csv(os.path.join(p['main20'], 'manifest.csv'))
        m = m[m['split'] == 'train']
        m['mirror'] = 0
    elif key == 'train_s1':
        f = os.path.join(p['s1'], 's1_main20_manifest.csv')
        m = pd.read_csv(f) if os.path.exists(f) else pd.concat([pd.read_csv(x) for x in sorted(glob.glob(os.path.join(p['s1'], 's1_main20_manifest_p*.csv')))])
    else:
        return {}
    return {r['clip']: dict(recording=r['recording'], start_frame=int(r['start_frame']), mirror=int(r['mirror'])) for _, r in m.iterrows()}


def build_meta(cfg, key, log=print):
    path = meta_path(cfg, key)
    if os.path.exists(path):
        log(f'[건너뜀] 라벨 캐시 있음: {path}')
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mem = list_members(cfg, key)
    man = _manifest(cfg, key)
    out = {}
    zips = {}
    for n, (z, d) in mem.items():
        zips.setdefault(z, []).append(n)
    for z, ns in zips.items():
        with zipfile.ZipFile(z) as zf:
            for n in ns:
                d = mem[n][1]
                rec = dict(mirror=0, recording=None, start_frame=None)
                rec.update(man.get(n, {}))
                if 'labels' in d:
                    lab = pd.read_csv(io.BytesIO(zf.read(d['labels'])))
                    rec['ev'] = lab[['frame', 'class', 'source', 'azimuth', 'distance', 'onscreen']].values.astype(np.int32)
                if 'traj' in d:
                    rec['psi'] = pd.read_csv(io.BytesIO(zf.read(d['traj'])))['psi'].values.astype(np.float32)
                out[n] = rec
    tmp = path + '.tmp'
    pickle.dump(out, open(tmp, 'wb')); os.replace(tmp, path)
    log(f'  {key}: 라벨 캐시 {len(out)}개 -> {path}')


def load_meta(cfg, key):
    return pickle.load(open(meta_path(cfg, key), 'rb'))


# ====================================================================== W1 소프트 onscreen 라벨
def soft_onscreen(G, az_w, el_w, yaw, pitch, tau):
    """생성기 onscreen 판정(dsgen.onscreen: x>0, |y|≤x·tanH, |z|≤x·tanV)과 같은 경계에서 잰 부드러운 라벨.
    카메라 좌표는 생성기 dsgen.to_cam 으로 구한다. 가로/세로 투영각 h=atan2(|y|,x), v=atan2(|z|,x) (도)로
    o_τ = σ((FOV_H/2 - h)/τ) · σ((atan(TAN_V) - v)/τ) · 1[x>0]."""
    import math
    c = G.to_cam(np.asarray(az_w, float), np.asarray(el_w, float), np.asarray(yaw, float), np.asarray(pitch, float))
    x, y, z = c[..., 0], c[..., 1], c[..., 2]
    h = np.degrees(np.arctan2(np.abs(y), x))
    v = np.degrees(np.arctan2(np.abs(z), np.maximum(x, 1e-9)))
    sig = lambda u: 1.0 / (1.0 + np.exp(-u))
    return (sig((G.FOV_H / 2 - h) / tau) * sig((math.degrees(math.atan(G.TAN_V)) - v) / tau) * (x > 0)).astype(np.float32)


def onsoft_path(cfg, key):
    return os.path.join(cfg['cache_root'], 'meta', f'{key}_onsoft_tau{float(cfg["on_soft_tau"]):g}.pkl')


def build_onsoft(cfg, key, log=print):
    """train 클립의 meta/{clip}_ext.csv(세계 방위각·고도, 카메라 yaw·pitch)로 소프트 onscreen 라벨을 만든다.
    {clip: (행 수,) float32} 이고 행 순서는 라벨 캐시의 사건 배열(ev)과 같다(확인함). 이진화(>0.5) 일치율을 로그에 남긴다."""
    from .config import import_dsgen
    path = onsoft_path(cfg, key)
    if os.path.exists(path):
        log(f'[건너뜀] 소프트 onscreen 캐시 있음: {path}')
        return
    G = import_dsgen(cfg, os.path.join(cfg['local_root'], 'raw'))
    meta = load_meta(cfg, key)
    mem = list_members(cfg, key)
    zips = {}
    for n, (z, d) in mem.items():
        zips.setdefault(z, []).append(n)
    out, agree, tot = {}, 0, 0
    for z, ns in zips.items():
        with zipfile.ZipFile(z) as zf:
            for n in ns:
                ext_m = mem[n][1]['labels'].replace('/labels/', '/meta/').replace('.csv', '_ext.csv')
                e = pd.read_csv(io.BytesIO(zf.read(ext_m)))
                ev = meta[n]['ev']
                if len(e) != len(ev) or not (e[['frame', 'class', 'source', 'azimuth', 'distance', 'onscreen']].values == ev).all():
                    raise RuntimeError(f'{n}: ext csv 행이 라벨과 다릅니다')
                s = soft_onscreen(G, e.az_world.values, e.el_world.values, e.cam_yaw.values, e.cam_pitch.values, float(cfg['on_soft_tau']))
                agree += int(((s > 0.5).astype(int) == ev[:, 5]).sum()); tot += len(ev)
                out[n] = s
    tmp = path + '.tmp'
    pickle.dump(out, open(tmp, 'wb')); os.replace(tmp, path)
    log(f'  {key}: 소프트 onscreen {len(out)}클립 -> {path}  (0.5 이진화 = 하드 라벨 {100 * agree / max(1, tot):.3f}%, {tot}행)')


def load_onsoft(cfg, key):
    return pickle.load(open(onsoft_path(cfg, key), 'rb'))


def extract_labels(cfg, split, log=print):
    """채점용 정답 csv 를 로컬에 푼다. split = val (val.zip) | test (private_test.zip, 최종 보고용)."""
    p = fd_paths(cfg)
    if split == 'val':
        z, pre, dst = os.path.join(p['main20'], 'val.zip'), 'val/labels/', os.path.join(cfg['local_root'], 'data', 'val', 'labels')
    else:
        z, pre, dst = os.path.join(p['main20'], 'private_test.zip'), 'private/test_labels/', os.path.join(cfg['local_root'], 'data', 'test', 'labels')
    if os.path.isdir(dst) and len(os.listdir(dst)) > 0:
        return dst
    os.makedirs(dst, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        for m in zf.namelist():
            if m.startswith(pre) and m.endswith('.csv'):
                open(os.path.join(dst, os.path.basename(m)), 'wb').write(zf.read(m))
    log(f'  {split} 정답 {len(os.listdir(dst))}개 -> {dst}')
    return dst


def prepare_pre(cfg, keys, log=print):
    log('오디오 캐시 예상 용량:')
    size_report(cfg, keys, log)
    for k in keys:
        build_meta(cfg, k, log)
        if cfg.get('loss_version') == 'w1' and cfg['stage'] == 2 and k.startswith('train'):
            build_onsoft(cfg, k, log)
        build_audio(cfg, k, log)
    extract_labels(cfg, 'val', log)
