"""
onthefly.py -- STARSS23 원본 FOA 를 학습 중에 바로 스테레오로 바꾸는 즉석 변환 로더 (CLAUDE.md 4.1).

원칙: 궤적, FOA->스테레오 변환, 라벨/onscreen 변환은 생성기(for_dataset/build/dsgen.py)의 함수를 import 해서 그대로 쓴다.
 - 궤적: dsgen.Traj.sample('sweep', 20, rng)       (main20 과 같은 분포)
 - 오디오: dsgen.foa_to_stereo(foa, -traj.psi(t))   (샘플 단위)
 - 라벨: dsgen.clip_labels(lab, start, 200, traj)    (방위각, onscreen, 프레임별 ψ)
 - 거르기: 라벨 프레임 30 미만(dsgen.MIN_LABELED), 피크 0.9999 초과(dsgen.PEAK_LIMIT) 창은 버리고 다시 뽑는다.
 - 생성기에는 게인 정규화가 없다(확인함). 그래서 여기도 없다.
 - 녹음 선택: 길이에 비례(build.py 의 train 클립 뽑기와 같음), 시작 프레임은 균등.
main20 train 방 7개의 녹음만 쓴다. val 방(9, 22)과 test 방(fold4)은 풀 때와 불러올 때 두 번 걸러 낸다.
이 파일은 numpy 만 쓴다(torch 없이 확인 가능).
"""
import os
import io
import json
import zipfile
import numpy as np
import pandas as pd
import soundfile as sf

from .config import import_dsgen, read_split_info, fd_paths, CLIP_FRAMES


def raw_dir(cfg):
    return os.path.join(cfg['local_root'], 'raw')


def _room(name):
    return int(name.split('_')[1][4:])


def _fold(name):
    return int(name.split('_')[0][4:])


def allowed_rooms(cfg):
    """train 방 목록. val/test 방과 겹치지 않는지 확인한다."""
    info = read_split_info(cfg)
    tr, va, te = set(info['train_rooms']), set(info['val_rooms']), set(info['test_rooms'])
    assert not (tr & va) and not (tr & te), "train 방이 val/test 방과 겹칩니다"
    return tr, va | te


# ====================================================================== 원본 풀기 (train 방만)
def prepare_raw(cfg, log=print):
    """STARSS23 foa_dev.zip, metadata_dev.zip 에서 train 방 녹음만 로컬에 푼다. 영상 유무는 video_dev.zip 목록으로 본다."""
    rd = raw_dir(cfg)
    marker = os.path.join(rd, 'recordings.json')
    if os.path.exists(marker):
        log(f'[건너뜀] 이미 풀려 있음: {rd}')
        return json.load(open(marker, encoding='utf8'))
    train_rooms, banned = allowed_rooms(cfg)
    st = fd_paths(cfg)['starss']
    keep = lambda n: n.endswith(('.wav', '.csv')) and _fold(os.path.basename(n)) == 3 and _room(os.path.basename(n)) in train_rooms
    n_out = 0
    for z in ('metadata_dev.zip', 'foa_dev.zip'):
        with zipfile.ZipFile(os.path.join(st, z)) as zf:
            mem = [m for m in zf.namelist() if not m.endswith('/') and keep(m)]
            for m in mem:
                assert _room(os.path.basename(m)) not in banned
                zf.extract(m, rd)
            n_out += len(mem)
            log(f'  {z}: train 방 파일 {len(mem)}개 풂')
    with zipfile.ZipFile(os.path.join(st, 'video_dev.zip')) as zf:
        has_video = {os.path.splitext(os.path.basename(m))[0] for m in zf.namelist() if m.endswith('.mp4')}
    recs = scan_raw(rd, has_video)
    json.dump(recs, open(marker, 'w', encoding='utf8'), indent=1)
    log(f'  녹음 {len(recs)}개 (영상 없음 {sum(not r["has_video"] for r in recs)}개), 방 {sorted({r["room"] for r in recs})}')
    return recs


def scan_raw(rd, has_video=None):
    """풀린 원본 폴더({rd}/metadata_dev, {rd}/foa_dev)의 녹음 목록. has_video 가 없으면 {rd}/video_dev 폴더를 본다."""
    if has_video is None:
        has_video = {os.path.splitext(os.path.basename(f))[0] for f in _glob(os.path.join(rd, 'video_dev'), '.mp4')}
    recs = []
    for c in sorted(_glob(os.path.join(rd, 'metadata_dev'), '.csv')):
        name = os.path.splitext(os.path.basename(c))[0]
        sub = os.path.basename(os.path.dirname(c))
        wav = os.path.join(rd, 'foa_dev', sub, name + '.wav')
        if not os.path.exists(wav):
            continue
        i = sf.info(wav)
        recs.append(dict(name=name, room=_room(name), csv=c, wav=wav, has_video=name in has_video,
                         audio_frames=int(np.floor(i.frames / i.samplerate * 10 + 1e-9)), subtype=i.subtype, sr=i.samplerate))
    return recs


def _glob(root, ext):
    out = []
    for r, _, fs in os.walk(root):
        out += [os.path.join(r, f) for f in fs if f.endswith(ext)]
    return out


# ====================================================================== 녹음 묶음 (메모리)
class RecordingBank:
    """train 방 녹음의 FOA(int16)와 라벨을 메모리에 올린다. 약 2~3GB."""
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        rd = raw_dir(cfg)
        js = os.path.join(rd, 'recordings.json')
        recs = json.load(open(js, encoding='utf8')) if os.path.exists(js) else scan_raw(rd)
        self.G = import_dsgen(cfg, rd)
        train_rooms, banned = allowed_rooms(cfg)
        recs = [r for r in recs if r['room'] in train_rooms and r['room'] not in banned]       # 한 번 더 거른다
        if not cfg.get('otf_include_no_video', False):
            recs = [r for r in recs if r['has_video']]                                         # 생성기와 같이 영상 없는 녹음 제외
        recs = [r for r in recs if r['audio_frames'] >= CLIP_FRAMES]
        self.recs, self.foa, self.labs, self.lab_cum = [], [], [], []
        for r in recs:
            x, sr = sf.read(r['wav'], dtype='int16' if r['subtype'] == 'PCM_16' else 'float32', always_2d=True)
            assert sr == self.G.SR and x.shape[1] == 4
            lab = self.G.load_labels(r['csv'])
            L = r['audio_frames']
            lf = np.zeros(L, np.int32)
            fr = lab.frame.values
            lf[np.unique(fr[fr < L])] = 1
            self.recs.append(r)
            self.foa.append(np.ascontiguousarray(x.T))           # (4, N)  W, Y, Z, X
            self.labs.append(lab)
            self.lab_cum.append(np.concatenate([[0], np.cumsum(lf)]))
        w = np.array([r['audio_frames'] for r in self.recs], float)
        self.weights = w / w.sum()
        self.index = {r['name']: i for i, r in enumerate(self.recs)}
        gb = sum(f.nbytes for f in self.foa) / 2 ** 30
        log(f'[즉석 변환] 녹음 {len(self.recs)}개, 방 {sorted({r["room"] for r in self.recs})}, '
            f'{sum(r["audio_frames"] for r in self.recs) / 36000:.2f}시간, FOA 메모리 {gb:.2f}GB')

    def foa_window(self, i, start, n=CLIP_FRAMES):
        """dsgen.read_foa 와 같은 구간을 float64 로 (sf.read dtype=float64 와 같은 값)."""
        a = start * self.G.SR // 10
        b = (start + n) * self.G.SR // 10
        x = self.foa[i][:, a:b]
        return x.astype(np.float64) / 32768.0 if x.dtype == np.int16 else x.astype(np.float64)

    def make(self, i, start, traj, n=CLIP_FRAMES):
        """녹음 i, 시작 프레임, 궤적으로 (스테레오 (2,N) float64, 공개 라벨 df, 궤적 df)를 만든다. 생성기 함수 그대로."""
        G = self.G
        foa = self.foa_window(i, start, n)
        t = np.arange(foa.shape[1]) / G.SR
        stereo = G.foa_to_stereo(foa, -traj.psi(t))
        pub, ext, trj = G.clip_labels(self.labs[i], start, n, traj)
        return stereo, pub, trj

    def sample(self, rng, n=CLIP_FRAMES, max_tries=100):
        """생성기 분포로 창 하나를 뽑는다. 라벨 30프레임 미만, 피크 초과는 다시 뽑는다."""
        G = self.G
        for _ in range(max_tries):
            i = int(rng.choice(len(self.recs), p=self.weights))
            L = self.recs[i]['audio_frames']
            start = int(rng.integers(0, L - n + 1))
            traj = G.Traj.sample('sweep', n / 10, rng)
            cum = self.lab_cum[i]
            if cum[start + n] - cum[start] < G.MIN_LABELED:
                continue
            stereo, pub, trj = self.make(i, start, traj, n)
            if np.abs(stereo).max() > G.PEAK_LIMIT:
                continue
            return dict(rec=i, start=start, traj=traj.params(), stereo=stereo, pub=pub, psi=trj.psi.values.astype(np.float64))
        raise RuntimeError('조건에 맞는 창을 찾지 못했습니다')


# ====================================================================== 필수 확인 1회: 생성기 출력과 일치
def check_match(cfg, n_clips=2, log=print):
    """main20 train 클립과 같은 녹음, 같은 구간, 같은 궤적으로 즉석 변환해 생성기 출력과 비교한다.
    오디오는 같은 방식(soundfile PCM_16)으로 양자화한 뒤 비트 단위로, 라벨과 궤적은 표 전체를 비교한다."""
    bank = RecordingBank(cfg, log)
    G = bank.G
    m = pd.read_csv(os.path.join(fd_paths(cfg)['main20'], 'manifest.csv'))
    m = m[(m['split'] == 'train') & (m.get('dropped', 0) == 0) & m['recording'].isin(bank.index)]
    rows = m.sample(n_clips, random_state=0)
    base = fd_paths(cfg)['main20']
    out = []
    for _, r in rows.iterrows():
        clip = r['clip']
        tr = G.Traj.from_params(20.0, {k: r[k] for k in ('kind', 'psi0', 's', 'omega', 'P', 'phi0')})
        stereo, pub, trj = bank.make(bank.index[r['recording']], int(r['start_frame']), tr)
        ref_wav, ref_lab, ref_trj = _read_main20_train(base, clip)
        buf = io.BytesIO()
        sf.write(buf, stereo.T, G.SR, format='WAV', subtype='PCM_16'); buf.seek(0)
        q, _ = sf.read(buf, dtype='int16', always_2d=True)
        res = dict(clip=clip, recording=r['recording'], start=int(r['start_frame']),
                   audio_bit_equal=bool(q.shape == ref_wav.shape and (q == ref_wav).all()),
                   audio_max_abs_err=float(np.abs(stereo.T - ref_wav / 32768.0).max()),
                   labels_equal=bool(pub.reset_index(drop=True).astype(float).equals(ref_lab.astype(float))),
                   traj_equal=bool(np.allclose(trj[['psi', 'phi']].values, ref_trj[['psi', 'phi']].values, atol=1e-9)))
        res['PASS'] = res['audio_bit_equal'] and res['labels_equal'] and res['traj_equal']
        out.append(res)
        log(json.dumps(res, ensure_ascii=False))
    return out


def _read_main20_train(base, clip):
    """main20 train 클립(풀린 폴더가 있으면 거기서, 없으면 train.zip 에서) 오디오(int16), 라벨, 궤적."""
    d = os.path.join(base, 'train')
    if os.path.exists(os.path.join(d, 'audio', clip + '.wav')):
        w, _ = sf.read(os.path.join(d, 'audio', clip + '.wav'), dtype='int16', always_2d=True)
        lab = pd.read_csv(os.path.join(d, 'labels', clip + '.csv'))
        trj = pd.read_csv(os.path.join(d, 'meta', clip + '_traj.csv'))
        return w, lab, trj
    with zipfile.ZipFile(os.path.join(base, 'train.zip')) as zf:
        w, _ = sf.read(io.BytesIO(zf.read(f'train/audio/{clip}.wav')), dtype='int16', always_2d=True)
        lab = pd.read_csv(io.BytesIO(zf.read(f'train/labels/{clip}.csv')))
        trj = pd.read_csv(io.BytesIO(zf.read(f'train/meta/{clip}_traj.csv')))
    return w, lab, trj
