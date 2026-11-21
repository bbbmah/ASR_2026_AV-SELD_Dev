"""
data.py -- 학습/평가용 데이터셋.

공통 출력(dict): audio (2, 480000) float32, clap (200, 512) float32, [labels (200, 6, 5, 13), dpsi (200,)], [video (200, 24, 77) fp16], name
 - 라벨 텐서는 베이스라인 utils.process_labels_adpit 로 만든다(정답 트랙 복제 방식 그대로).
 - 좌우 반전: 음향 L/R 교환, 방위각 θ -> -θ, Δψ -> -Δψ. 앞/뒤, 거리, onscreen 은 그대로. 영상은 반전 캐시(있을 때만).
 - 학습용 데이터셋은 길이 = steps_per_epoch × batch 이고, 항목 i 의 난수는 (seed, epoch, i) 로 정해진다(이어 하기 때 재현).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CLIP_FRAMES, NB_CLASSES


def wrap180(x):
    return (np.asarray(x, np.float64) + 180.0) % 360.0 - 180.0


def dpsi_from_psi(psi):
    """프레임별 ψ(도) -> Δψ_t = ψ_t - ψ_{t-1} (첫 프레임은 ψ_1 - ψ_0), -180~180 로 감는다."""
    d = wrap180(np.diff(np.asarray(psi, np.float64)))
    return np.concatenate([d[:1], d]).astype(np.float32)


def events_to_adpit(ev, flip=False, on_soft=None):
    """ev: (n, 6) [frame, class, source, azimuth, distance(cm), onscreen] -> (200, 6, 5, 13) 텐서
    on_soft: (n,) 이면 onscreen 자리에 이 값(W1 소프트 라벨)을 넣는다. 좌우 반전에도 그대로다(시야가 좌우 대칭)."""
    import utils                                                   # 베이스라인 seld_challenge/utils.py
    d = {}
    for i, (fr, cl, src, az, dist, on) in enumerate(ev):
        if fr >= CLIP_FRAMES:
            continue
        a = np.radians(-az if flip else az)
        o = int(on) if on_soft is None else float(on_soft[i])
        d.setdefault(int(fr), []).append([int(cl), int(src), float(np.cos(a)), float(np.sin(a)), float(dist), o])
    return utils.process_labels_adpit(d, CLIP_FRAMES, NB_CLASSES)


def pub_to_events(pub):
    return pub[['frame', 'class', 'source', 'azimuth', 'distance', 'onscreen']].values.astype(np.int32)


class TrainBase(Dataset):
    def __init__(self, cfg, length):
        self.cfg = cfg
        self.length = int(length)
        self.epoch = 0
        self.seed = int(cfg['seed'])

    def set_epoch(self, e):
        self.epoch = int(e)

    def rng(self, i):
        return np.random.default_rng([self.seed, self.epoch, int(i)])

    def __len__(self):
        return self.length


class PreTrainDataset(TrainBase):
    """미리 렌더링한 클립(main20 train S0, S1)을 무작위로(복원 추출) 뽑는다."""
    def __init__(self, cfg, length, keys, clap, video=None):
        super().__init__(cfg, length)
        from .prerender import load_audio, load_meta, load_onsoft
        self.items = []                                             # (key, 인덱스, 이름)
        self.audio, self.meta, self.onsoft = {}, {}, {}
        use_soft = cfg.get('loss_version') == 'w1' and cfg['stage'] == 2 and cfg['loss_onscreen']
        for k in keys:
            names, arr = load_audio(cfg, k)
            self.audio[k] = arr
            self.meta[k] = load_meta(cfg, k)
            if use_soft:
                self.onsoft[k] = load_onsoft(cfg, k)                # W1 소프트 onscreen (prepare_pre 가 만든다)
            self.items += [(k, i, n) for i, n in enumerate(names)]
        self.clap = clap
        self.video = video                                          # VideoStore 또는 None
        if video is not None:
            self.items = [x for x in self.items if video.has(x[2])]
        self.flip_prob = float(cfg['flip_prob'])
        self.video_drop = float(cfg['video_drop']) if (video is not None and cfg.get('loss_version') == 'w1') else 0.0

    def __getitem__(self, i):
        rng = self.rng(i)
        k, j, n = self.items[int(rng.integers(len(self.items)))]
        m = self.meta[k][n]
        flip = rng.random() < self.flip_prob
        if self.video is not None and flip and not self.video.has(n, flip=True):
            flip = False                                            # 반전 영상 캐시가 없는 클립(S1 거울 클립)은 뒤집지 않는다
        audio = self.audio[k][j].astype(np.float32) / 32768.0
        if flip:
            audio = audio[::-1].copy()
        clap = self.clap.by_rec(m['recording'], m['start_frame'])
        d = dict(audio=torch.from_numpy(audio), clap=torch.from_numpy(clap), name=n,
                 labels=events_to_adpit(m['ev'], flip, self.onsoft[k][n] if k in self.onsoft else None), dpsi=torch.from_numpy(dpsi_from_psi(m['psi']) * (-1 if flip else 1)))
        if self.video is not None:
            d['video'] = self.video.get(n, flip)
            if self.video_drop > 0:                                 # W1 영상 드롭아웃: 0 이면 이 샘플의 영상 특징 전체를 0
                d['vkeep'] = torch.tensor(float(rng.random() >= self.video_drop))
        return d


class OTFTrainDataset(TrainBase):
    """즉석 변환: 매번 원본 녹음에서 새 창과 새 궤적을 뽑는다."""
    def __init__(self, cfg, length, bank, clap):
        super().__init__(cfg, length)
        self.bank, self.clap = bank, clap
        self.flip_prob = float(cfg['flip_prob'])

    def __getitem__(self, i):
        rng = self.rng(i)
        s = self.bank.sample(rng)
        flip = rng.random() < self.flip_prob
        audio = s['stereo'].astype(np.float32)
        if flip:
            audio = audio[::-1].copy()
        rec = self.bank.recs[s['rec']]['name']
        clap = self.clap.by_rec(rec, s['start'])
        return dict(audio=torch.from_numpy(audio), clap=torch.from_numpy(clap), name=f'{rec}@{s["start"]}',
                    labels=events_to_adpit(pub_to_events(s['pub']), flip),
                    dpsi=torch.from_numpy(dpsi_from_psi(s['psi']) * (-1 if flip else 1)))


class EvalDataset(Dataset):
    """val/test 클립 (반전 없음, 순서대로)."""
    def __init__(self, cfg, key, clap, video=None, with_labels=True):
        from .prerender import load_audio, load_meta
        self.names, self.arr = load_audio(cfg, key)
        self.meta = load_meta(cfg, key) if with_labels else None
        self.clap, self.video, self.with_labels = clap, video, with_labels

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        d = dict(audio=torch.from_numpy(self.arr[i].astype(np.float32) / 32768.0), clap=torch.from_numpy(self.clap.by_clip(n)), name=n)
        if self.with_labels:
            m = self.meta[n]
            d['labels'] = events_to_adpit(m['ev'])
            d['dpsi'] = torch.from_numpy(dpsi_from_psi(m['psi']))
        if self.video is not None:
            d['video'] = self.video.get(n, False)
        return d


class VideoStore:
    """OWL-ViT 열 특징 캐시를 메모리에 올린다. {이름: (묶음 텐서, 인덱스)}, 반전은 따로."""
    def __init__(self, cfg, keys, log=print):
        from .visual import load_shards, flip_keys
        self.data, self.flip = {}, {}
        tot = 0
        for k in keys:
            names, x = load_shards(cfg, k)
            for i, n in enumerate(names):
                self.data[n] = (x, i)
            tot += x.numel() * 2
            if k.startswith('train'):
                try:
                    fn, fx = load_shards(cfg, flip_keys(k))
                    for i, n in enumerate(fn):
                        self.flip[n] = (fx, i)
                    tot += fx.numel() * 2
                except FileNotFoundError:
                    log(f'  [참고] 반전 영상 캐시 없음: {k}')
        log(f'  영상 특징 {len(self.data)}개 (+반전 {len(self.flip)}개), 메모리 {tot / 2 ** 30:.1f}GB')

    def has(self, name, flip=False):
        return name in (self.flip if flip else self.data)

    def get(self, name, flip=False):
        x, i = (self.flip if flip else self.data)[name]
        return x[i]
