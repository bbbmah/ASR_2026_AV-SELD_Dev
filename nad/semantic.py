"""
semantic.py -- (B) 의미 경로: Mid 신호를 48kHz 로 바꿔 고정 LAION-CLAP 에 넣고, 2초 창을 0.5초씩 밀며 512차원 벡터를 뽑아 캐시한다.

 - Mid = (L+R)/2 = W 이다(생성기 변환 L=W+newY, R=W-newY). 그래서 Mid 는 카메라 궤적과 좌우 반전에 영향을 받지 않는다.
 - val/test: 클립마다 창 37개(시작 0, 0.5, ..., 18초). 캐시 {cache_root}/clap/{val,test}.pt
 - train(미리 렌더링, 즉석 변환 둘 다): 원본 녹음의 W 채널에서 녹음 단위로 0.5초 간격 창을 한 번 뽑아 둔다
   ({cache_root}/clap/rec_train.pt). 클립의 시작 프레임으로 같은 녹음 구간을 잘라 쓴다. Mid=W 라 같은 신호이고,
   클립마다 따로 뽑는 것(약 33만 창)보다 훨씬 싸다. 클립 가장자리 1초 안쪽만 창이 클립 밖까지 걸치는 점이 다르다.
 - 프레임(0.1k+0.05초)마다 창 가운데 시각 기준 선형 보간으로 200×512 를 만든다(모델 안에서 256차원으로 사영).
"""
import os
import json
import time
import numpy as np
import torch
from scipy.signal import resample_poly

from .config import CLIP_FRAMES


def clap_dir(cfg):
    return os.path.join(cfg['cache_root'], 'clap')


def frame_times(start_frame=0, n=CLIP_FRAMES):
    return (start_frame + np.arange(n)) * 0.1 + 0.05


def interp_frames(emb, centers, times):
    """emb (K, D), centers (K,) 초, times (T,) 초 -> (T, D). 범위 밖은 끝값."""
    t = np.clip(times, centers[0], centers[-1])
    j = np.clip(np.searchsorted(centers, t, side='right') - 1, 0, len(centers) - 2)
    w = ((t - centers[j]) / (centers[j + 1] - centers[j]))[:, None]
    e = emb.astype(np.float32)
    return (1 - w) * e[j] + w * e[j + 1]


class ClapEncoder:
    def __init__(self, cfg, device):
        from transformers import ClapModel, ClapProcessor
        self.proc = ClapProcessor.from_pretrained(cfg['clap_model'])
        self.model = ClapModel.from_pretrained(cfg['clap_model']).to(device).eval()
        self.device = device
        self.sr = self.proc.feature_extractor.sampling_rate        # 48000
        self.win = int(cfg['clap_win_s'] * self.sr)

    @torch.no_grad()
    def windows(self, x48, hop_s):
        """x48: 48kHz 1채널. 창 시작 0, hop, ... (끝이 넘지 않게). 반환 (K, 512) float16, 가운데 시각 (K,)."""
        hop = int(hop_s * self.sr)
        starts = np.arange(0, len(x48) - self.win + 1, hop)
        out = []
        for i in range(0, len(starts), 64):
            segs = [x48[s:s + self.win].astype(np.float32) for s in starts[i:i + 64]]
            inp = self.proc.feature_extractor(segs, sampling_rate=self.sr, return_tensors='pt')
            inp = {k: v.to(self.device) for k, v in inp.items() if k in ('input_features', 'is_longer')}
            e = self.model.get_audio_features(**inp)
            if not torch.is_tensor(e):                                  # transformers 5 는 출력 객체를 돌려준다
                cand = [getattr(e, k, None) for k in ('audio_embeds', 'pooler_output')]
                cand = [c for c in cand if torch.is_tensor(c) and c.dim() == 2 and c.shape[-1] == 512]
                if not cand:
                    raise RuntimeError(f'CLAP 출력에서 512차원 임베딩을 찾지 못했습니다: {type(e).__name__}')
                e = cand[0]
            assert e.shape[-1] == 512, f'CLAP 출력 차원이 512 가 아닙니다: {tuple(e.shape)} (transformers 버전 확인)'
            out.append(e.float().cpu())
        centers = (starts + self.win / 2) / self.sr
        return torch.cat(out).half().numpy(), centers


def to48(x24):
    return resample_poly(np.asarray(x24, np.float64), 2, 1)


def build_rec_train(cfg, device='cuda', log=print):
    """train 방 원본 녹음의 W 채널로 녹음 단위 CLAP 캐시."""
    import soundfile as sf
    from .onthefly import raw_dir, allowed_rooms
    path = os.path.join(clap_dir(cfg), 'rec_train.pt')
    if os.path.exists(path):
        log(f'[건너뜀] {path}')
        return
    os.makedirs(clap_dir(cfg), exist_ok=True)
    recs = json.load(open(os.path.join(raw_dir(cfg), 'recordings.json'), encoding='utf8'))
    train_rooms, banned = allowed_rooms(cfg)
    enc = ClapEncoder(cfg, device)
    out, t0 = {}, time.time()
    for i, r in enumerate(recs):
        assert r['room'] in train_rooms and r['room'] not in banned
        w, sr = sf.read(r['wav'], dtype='float64', always_2d=True)
        emb, centers = enc.windows(to48(w[:, 0]), cfg['otf_clap_hop_s'])
        out[r['name']] = dict(emb=emb, centers=centers.astype(np.float32))
        if (i + 1) % 10 == 0:
            log(f'  CLAP 녹음 {i + 1}/{len(recs)} ({time.time() - t0:.0f}초)')
    torch.save(out, path + '.tmp'); os.replace(path + '.tmp', path)
    log(f'  CLAP 녹음 캐시 {len(out)}개 -> {path} ({time.time() - t0:.0f}초)')


def build_clips(cfg, key, device='cuda', log=print):
    """val/test 클립의 Mid 로 클립 단위 CLAP 캐시."""
    from .prerender import load_audio
    path = os.path.join(clap_dir(cfg), f'{key}.pt')
    if os.path.exists(path):
        log(f'[건너뜀] {path}')
        return
    os.makedirs(clap_dir(cfg), exist_ok=True)
    names, arr = load_audio(cfg, key)
    enc = ClapEncoder(cfg, device)
    embs, t0, centers = [], time.time(), None
    for i, n in enumerate(names):
        mid = (arr[i, 0].astype(np.float64) + arr[i, 1].astype(np.float64)) / 2 / 32768.0
        e, centers = enc.windows(to48(mid), cfg['clap_hop_s'])
        embs.append(e)
        if (i + 1) % 100 == 0:
            log(f'  CLAP {key} {i + 1}/{len(names)} ({time.time() - t0:.0f}초)')
    torch.save(dict(names=names, emb=np.stack(embs), centers=centers.astype(np.float32)), path + '.tmp')
    os.replace(path + '.tmp', path)
    log(f'  CLAP {key} {len(names)}개 -> {path}')


class ClapLookup:
    """클립 이름 또는 (녹음, 시작 프레임)으로 200×512 프레임 특징을 돌려준다."""
    def __init__(self, cfg, keys):
        self.rec, self.clip = None, {}
        if any(k.startswith('train') for k in keys) or 'rec' in keys:
            self.rec = torch.load(os.path.join(clap_dir(cfg), 'rec_train.pt'), weights_only=False)
        for k in keys:
            if k in ('val', 'test'):
                d = torch.load(os.path.join(clap_dir(cfg), f'{k}.pt'), weights_only=False)
                for i, n in enumerate(d['names']):
                    self.clip[n] = (d['emb'][i], d['centers'])

    def by_clip(self, name):
        e, c = self.clip[name]
        return interp_frames(e, c, frame_times())

    def by_rec(self, rec, start_frame):
        d = self.rec[rec]
        return interp_frames(d['emb'], d['centers'], frame_times(start_frame))
