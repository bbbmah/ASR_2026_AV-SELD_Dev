"""
visual.py -- (C) 영상 경로: 고정 OWL-ViT(ViT-B/32)로 프레임마다 패치 24×24 의 클래스 점수(13)와 임베딩(768)을 구하고,
열(24개) 단위로 모아 클립당 200×24×77 (fp16) 특징을 캐시한다.

 - 클래스 점수: configs/owl_prompts.yaml 의 13개 문장과 패치별 로짓(시그모이드 전). 열 안에서 max.
 - 임베딩: 768차원 패치 임베딩(image_embeds)을 PCA 64차원으로. PCA 는 S0 train 프레임에서만 학습. 열 안에서 mean
   (PCA 는 선형이므로 열 평균 후 사영해도 같다).
 - 열 방위각: f = 320/tan(50°), θ_j = atan((320 - x_j)/f), x_j = 원본 640px 기준 열 j 의 가운데 x.
   OWL-ViT 전처리가 정사각형으로 늘리든(stretch) 아래/오른쪽에 여백을 붙이든(pad) 가로는 640px 전체가 24열이 되므로
   x_j = (j+0.5)·640/24 로 같다(여백이면 세로 행만 일부 버린다). 어느 쪽인지는 check() 가 processor 설정을 읽어 정한다.
 - 좌우 반전 증강용으로 뒤집은 영상의 특징도 따로 캐시한다(거울이 아닌 train 클립만: S1 의 거울 클립은 이미 반전이므로 다시 뒤집지 않는다).
 - 캐시 위치: {cache_root}/owl_{해시}/{key}_{묶음:03d}.pt  ({'names', 'data'})  + meta.json (열 방위각, 표준화 통계 등)
"""
import os
import io
import json
import math
import time
import hashlib
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .config import CLIP_FRAMES, NB_CLASSES

NCOL = 24
VDIM = NB_CLASSES + 64
SHARD = 250
W_IMG, H_IMG = 640, 360


def prompts(cfg):
    p = yaml.safe_load(open(cfg['owl_prompts'], encoding='utf8'))['prompts']
    assert len(p) == NB_CLASSES, f'문장이 {NB_CLASSES}개여야 합니다: {len(p)}'
    return p


def owl_dir(cfg):
    h = hashlib.sha1(json.dumps([cfg['owl_model'], prompts(cfg), cfg['owl_pca_dim']]).encode()).hexdigest()[:8]
    return os.path.join(cfg['cache_root'], f'owl_{h}')


def column_azimuths_deg():
    f = 320.0 / math.tan(math.radians(50.0))
    x = (np.arange(NCOL) + 0.5) * W_IMG / NCOL
    return np.degrees(np.arctan((320.0 - x) / f))


def flip_keys(key):
    return f'{key}_flip'


def size_report(cfg, keys, log=print):
    from .prerender import list_members, load_meta
    per = CLIP_FRAMES * NCOL * VDIM * 2 / 2 ** 20
    tot = 0.0
    for k in keys:
        n = len(list_members(cfg, k))
        nf = 0
        if k.startswith('train'):
            try:
                meta = load_meta(cfg, k); nf = sum(1 for v in meta.values() if not v.get('mirror', 0))
            except FileNotFoundError:
                nf = n
        mb = (n + nf) * per
        tot += mb
        log(f'  {k}: 클립 {n}개 + 반전 {nf}개 -> {mb / 1024:.2f}GB')
    log(f'  클립당 {per:.2f}MB, 합계 약 {tot / 1024:.1f}GB (드라이브 {owl_dir(cfg)})')
    return tot


class OwlEncoder:
    def __init__(self, cfg, device):
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        self.proc = OwlViTProcessor.from_pretrained(cfg['owl_model'])
        self.model = OwlViTForObjectDetection.from_pretrained(cfg['owl_model']).to(device).eval()
        self.device = device
        t = self.proc(text=[prompts(cfg)], return_tensors='pt')
        self.ids, self.am = t['input_ids'].to(device), t['attention_mask'].to(device)
        ip = self.proc.image_processor
        sz = ip.size
        self.side = int(sz['height'] if isinstance(sz, dict) and 'height' in sz else (sz['shortest_edge'] if isinstance(sz, dict) else sz))
        self.mode = 'pad' if getattr(ip, 'do_pad', False) else 'stretch'
        crop = getattr(ip, 'crop_size', None)
        if getattr(ip, 'do_center_crop', False) and crop is not None:
            cs = crop['height'] if isinstance(crop, dict) else crop
            assert int(cs) >= self.side, f'가운데 자르기({cs})가 있어 열 위치 계산이 달라집니다. 코드를 고쳐야 합니다'
        self.mean = torch.tensor(ip.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=device).view(1, 3, 1, 1)
        g = self.side // 32
        assert g == NCOL, f'패치 격자가 {g}x{g} 입니다 (24 를 가정)'
        self.valid_rows = NCOL if self.mode == 'stretch' else int(math.ceil(H_IMG / W_IMG * NCOL))

    def describe(self):
        ip = self.proc.image_processor
        return dict(mode=self.mode, side=self.side, valid_rows=self.valid_rows, do_resize=getattr(ip, 'do_resize', None),
                    size=getattr(ip, 'size', None), do_center_crop=getattr(ip, 'do_center_crop', None), crop_size=getattr(ip, 'crop_size', None),
                    do_pad=getattr(ip, 'do_pad', None), resample=str(getattr(ip, 'resample', None)))

    def preprocess(self, frames):
        """frames: (N, 360, 640, 3) uint8 RGB -> (N, 3, side, side)"""
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device).permute(0, 3, 1, 2).float()
        if self.mode == 'pad':
            x = F.pad(x, (0, 0, 0, W_IMG - H_IMG), value=0.0)        # 아래에 여백 -> 640x640 (OWLv2 식)
        x = F.interpolate(x, size=(self.side, self.side), mode='bicubic', align_corners=False, antialias=True)
        return ((x.clamp(0, 255) / 255.0) - self.mean) / self.std

    def check(self, frame, log=print):
        """processor 설정과, 직접 만든 전처리와 processor 출력의 차이를 본다(프레임 1장)."""
        from PIL import Image
        ref = self.proc(images=Image.fromarray(frame), return_tensors='pt')['pixel_values'].to(self.device)
        ours = self.preprocess(frame[None])
        d = self.describe(); d['preprocess_max_abs_diff'] = float((ref - ours).abs().max()); d['preprocess_mean_abs_diff'] = float((ref - ours).abs().mean())
        d['column_azimuth_deg'] = [round(float(a), 2) for a in column_azimuths_deg()]
        log(json.dumps(d, ensure_ascii=False, default=str))
        return d

    @torch.no_grad()
    def run(self, frames):
        """-> 로짓 (N,24,24,13), 패치 임베딩 (N,24,24,768)  [행(y), 열(x)]"""
        px = self.preprocess(frames)
        N = px.shape[0]
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.device.startswith('cuda')):
            out = self.model(input_ids=self.ids.repeat(N, 1), attention_mask=self.am.repeat(N, 1), pixel_values=px)
        return out.logits.float().view(N, NCOL, NCOL, NB_CLASSES), out.image_embeds.float()


def _read_frames(zip_path, member, tmpdir):
    import cv2
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(member)
    fd, tmp = tempfile.mkstemp(suffix='.mp4', dir=tmpdir); os.close(fd)
    open(tmp, 'wb').write(data)
    cap = cv2.VideoCapture(tmp); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release(); os.remove(tmp)
    if not fr:
        raise RuntimeError(f'영상을 읽지 못함: {member}')
    while len(fr) < CLIP_FRAMES:
        fr.append(fr[-1])
    return np.stack(fr[:CLIP_FRAMES])


def fit_pca(cfg, enc, log=print):
    """S0 train 클립 일부의 프레임에서 패치 임베딩 PCA(768 -> 64)."""
    from .prerender import list_members
    d = owl_dir(cfg); path = os.path.join(d, 'pca.pt')
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    os.makedirs(d, exist_ok=True)
    mem = list_members(cfg, 'train_s0')
    names = sorted(mem)
    rng = np.random.default_rng(0)
    pick = [names[i] for i in rng.choice(len(names), min(cfg['owl_pca_clips'], len(names)), replace=False)]
    fidx = np.linspace(0, CLIP_FRAMES - 1, cfg['owl_pca_frames']).round().astype(int)
    s = torch.zeros(768, dtype=torch.float64, device=enc.device); ss = torch.zeros(768, 768, dtype=torch.float64, device=enc.device); n = 0
    tmpdir = os.path.join(cfg['local_root'], 'tmp'); os.makedirs(tmpdir, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        for i, fr in enumerate(ex.map(lambda nm: _read_frames(mem[nm][0], mem[nm][1]['video'], tmpdir), pick)):
            _, emb = enc.run(fr[fidx])
            e = emb[:, :enc.valid_rows].reshape(-1, 768).double()
            s += e.sum(0); ss += e.T @ e; n += e.shape[0]
            if (i + 1) % 50 == 0:
                log(f'  PCA 표본 {i + 1}/{len(pick)} ({time.time() - t0:.0f}초)')
    mu = s / n
    cov = ss / n - torch.outer(mu, mu)
    ev, evec = torch.linalg.eigh(cov)
    order = torch.argsort(ev, descending=True)[:cfg['owl_pca_dim']]
    out = dict(mean=mu.float().cpu(), W=evec[:, order].float().cpu(), explained=float(ev[order].sum() / ev.sum()), n_vectors=n, clips=len(pick))
    torch.save(out, path)
    log(f'  PCA: 벡터 {n}개, 64차원 설명 분산 {out["explained"]:.3f} -> {path}')
    return out


def clip_features(enc, frames, pca, batch):
    """(200, 360, 640, 3) -> (200, 24, 77) float16"""
    W, mu = pca['W'].to(enc.device), pca['mean'].to(enc.device)
    out = []
    for i in range(0, len(frames), batch):
        lg, emb = enc.run(frames[i:i + batch])
        v = enc.valid_rows
        cls = lg[:, :v].amax(dim=1)                                  # (n, 24, 13) 열 안 max
        e = (emb[:, :v].mean(dim=1) - mu) @ W                        # (n, 24, 64) 열 안 mean -> PCA
        out.append(torch.cat([cls, e], dim=-1).half().cpu())
    return torch.cat(out)


def build(cfg, keys, device='cuda', log=print):
    from .prerender import list_members, load_meta
    d = owl_dir(cfg); os.makedirs(d, exist_ok=True)
    enc = OwlEncoder(cfg, device)
    pca = fit_pca(cfg, enc, log)
    tmpdir = os.path.join(cfg['local_root'], 'tmp'); os.makedirs(tmpdir, exist_ok=True)
    for key in keys:
        mem = list_members(cfg, key)
        names = list(mem)
        meta = load_meta(cfg, key) if key.startswith('train') else {}
        flip_ok = {n for n in names if key.startswith('train') and not meta.get(n, {}).get('mirror', 0)}
        t0 = time.time()
        for s in range(math.ceil(len(names) / SHARD)):
            p = os.path.join(d, f'{key}_{s:03d}.pt'); pf = os.path.join(d, f'{flip_keys(key)}_{s:03d}.pt')
            chunk = names[s * SHARD:(s + 1) * SHARD]
            need_flip = any(n in flip_ok for n in chunk)
            if os.path.exists(p) and (os.path.exists(pf) or not need_flip):
                continue
            feats, ffeats, fnames = [], [], []
            with ThreadPoolExecutor(4) as ex:
                for n, fr in zip(chunk, ex.map(lambda nm: _read_frames(mem[nm][0], mem[nm][1]['video'], tmpdir), chunk)):
                    feats.append(clip_features(enc, fr, pca, cfg['owl_batch']))
                    if n in flip_ok:
                        ffeats.append(clip_features(enc, fr[:, :, ::-1], pca, cfg['owl_batch'])); fnames.append(n)
            torch.save(dict(names=chunk, data=torch.stack(feats)), p + '.tmp'); os.replace(p + '.tmp', p)
            if fnames:
                torch.save(dict(names=fnames, data=torch.stack(ffeats)), pf + '.tmp'); os.replace(pf + '.tmp', pf)
            log(f'  OWL {key} 묶음 {s}: {len(chunk)}개 (+반전 {len(fnames)}) ({time.time() - t0:.0f}초)')
    mp = os.path.join(d, 'meta.json')
    if not os.path.exists(mp):
        x = load_shards(cfg, 'train_s0', max_shards=2)[1].float()
        meta = dict(enc.describe(), col_az_deg=column_azimuths_deg().tolist(), prompts=prompts(cfg), model=cfg['owl_model'],
                    vis_mean=x.mean(dim=(0, 1, 2)).tolist(), vis_std=x.std(dim=(0, 1, 2)).clamp(min=1e-3).tolist(), pca_explained=pca['explained'])
        json.dump(meta, open(mp, 'w', encoding='utf8'), indent=1, default=str)
    log(f'OWL 캐시 완료: {d}')


def load_shards(cfg, key, max_shards=None):
    import glob
    fs = sorted(glob.glob(os.path.join(owl_dir(cfg), f'{key}_[0-9][0-9][0-9].pt')))
    if max_shards:
        fs = fs[:max_shards]
    if not fs:
        raise FileNotFoundError(f'OWL 캐시가 없습니다: {owl_dir(cfg)}/{key}_*.pt  (owl 단계를 먼저 실행)')
    names, parts = [], []
    for f in fs:
        dd = torch.load(f, weights_only=False)
        names += dd['names']; parts.append(dd['data'])
    return names, torch.cat(parts)


def load_meta(cfg):
    return json.load(open(os.path.join(owl_dir(cfg), 'meta.json'), encoding='utf8'))
