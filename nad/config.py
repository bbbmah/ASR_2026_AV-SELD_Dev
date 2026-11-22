"""
config.py -- configs/nad_base.yaml + 프리셋 + --set key=value 를 합쳐 설정(Config)을 만든다.
for_dataset 의 코드(생성기 dsgen, 베이스라인 채점/라벨 함수, NTU 특징 함수)를 읽기 전용으로 import 하는 함수도 여기 있다.
for_dataset 안에 파일이 생기지 않도록 바이트코드(__pycache__) 쓰기를 끈다.
"""
import os
import sys
import importlib.util

sys.dont_write_bytecode = True          # for_dataset 안에 __pycache__ 를 만들지 않는다
import yaml

NAD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_YAML = os.path.join(NAD_DIR, 'configs', 'nad_base.yaml')

# 프리셋: 개별 옵션이 비어 있을 때 쓰는 값
_W1 = dict(loss_version='w1', ema_decay=0.999, grad_clip=1.0)      # MODEL_DEV_2.md W1 설정
PRESETS = {
    'N1-pre': dict(stage=1, data_source='prerender', loss_fb=False, loss_onscreen=False, loss_rot=False, select_metric='F_basic'),
    'N1-otf': dict(stage=1, data_source='onthefly', loss_fb=False, loss_onscreen=False, loss_rot=False, select_metric='F_basic'),
    'N2':     dict(stage=2, data_source='prerender', loss_fb=True, loss_onscreen=True, loss_rot=True, select_metric='F_ext'),
    # W1 (MODEL_DEV_2.md): 새 손실(접은 각도 α, 스무딩, 소프트 onscreen, 정규화), EMA, 클리핑 1.0, 2단계 영상 드롭아웃
    'N1w':    dict(stage=1, data_source='onthefly', loss_fb=False, loss_onscreen=False, loss_rot=False, select_metric='F_basic', **_W1),
    'N1w-rb': dict(stage=1, data_source='onthefly', loss_fb=False, loss_onscreen=False, loss_rot=False, select_metric='F_basic',
                   otf_room_balance=True, **_W1),
    # ① 활성 항 세기만 바꾼 N1w (전체 칸 평균 정규화는 그대로, ①에 w_act 를 곱함. 순열 선택의 가중 합에도 같은 w_act)
    'N1w-a5':  dict(stage=1, data_source='onthefly', loss_fb=False, loss_onscreen=False, loss_rot=False, select_metric='F_basic',
                    w_act=5.0, **_W1),
    'N1w-a10': dict(stage=1, data_source='onthefly', loss_fb=False, loss_onscreen=False, loss_rot=False, select_metric='F_basic',
                    w_act=10.0, **_W1),
    'N2w':    dict(stage=2, data_source='prerender', loss_fb=True, loss_onscreen=True, loss_rot=True, select_metric='F_ext', **_W1),
}
PRESET_KEYS = ('data_source', 'loss_fb', 'loss_onscreen', 'loss_rot', 'select_metric')
FORCED_KEYS = ('loss_version', 'ema_decay', 'grad_clip', 'otf_room_balance', 'w_act')    # 프리셋에 있으면 --set 으로 주지 않은 한 프리셋 값

CLIP_FRAMES = 200
NB_CLASSES = 13
TRACKS = 3


class Config(dict):
    __getattr__ = dict.__getitem__


def _blank(v):
    return v is None or (isinstance(v, str) and v.strip() == '')


def parse_sets(sets):
    out = {}
    for item in sets or []:
        if '=' not in item:
            raise ValueError(f"--set 은 key=value 형태여야 합니다: {item}")
        k, v = item.split('=', 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def load_config(preset=None, sets=None, base_yaml=BASE_YAML):
    with open(base_yaml, encoding='utf8') as f:
        base = yaml.safe_load(f)
    cfg = dict(base)
    over = parse_sets(sets)
    if preset is not None:
        over['preset'] = preset
    unknown = [k for k in over if k not in base]
    if unknown:
        raise KeyError(f"알 수 없는 설정 이름: {unknown}  (configs/nad_base.yaml 참조)")
    cfg.update(over)
    if cfg['preset'] not in PRESETS:
        raise ValueError(f"preset 은 {list(PRESETS)} 중 하나: {cfg['preset']}")
    pv = PRESETS[cfg['preset']]
    cfg['stage'] = pv['stage']
    for k in PRESET_KEYS:
        if _blank(cfg.get(k)):
            cfg[k] = pv[k]
    for k in FORCED_KEYS:
        if k in pv and k not in over:
            cfg[k] = pv[k]
    if cfg['loss_version'] not in ('v0', 'w1'):
        raise ValueError("loss_version 은 v0 | w1")
    for k in ('loss_fb', 'loss_onscreen', 'loss_rot'):
        cfg[k] = bool(cfg[k])
    if cfg['data_source'] not in ('prerender', 'onthefly'):
        raise ValueError("data_source 는 prerender | onthefly")
    if cfg['stage'] == 2 and cfg['data_source'] != 'prerender':
        raise ValueError("2단계(N2)는 미리 렌더링한 데이터(영상 특징 캐시)로만 학습한다")
    cfg['train_stages'] = sorted({int(x) for x in str(cfg['train_stages']).replace(' ', '').split(',') if x != ''})

    if _blank(cfg['drive_root']):
        raise ValueError("drive_root 를 지정하세요 (예: --set drive_root='/content/drive/MyDrive/Colab Notebooks/ASR_2026-2')")
    dr = cfg['drive_root']
    cfg['for_dataset'] = os.path.join(dr, 'for_dataset')
    cfg['nad_dir'] = os.path.join(dr, 'New_Architecture_dev')
    cfg['out_root'] = cfg['out_root'] if not _blank(cfg['out_root']) else os.path.join(cfg['nad_dir'], 'runs')
    cfg['cache_root'] = cfg['cache_root'] if not _blank(cfg['cache_root']) else os.path.join(cfg['nad_dir'], 'cache')
    data = 'S' + ''.join(str(s) for s in cfg['train_stages']) if cfg['data_source'] == 'prerender' else 'otf'
    cfg['run_name'] = f"nad_{cfg['preset']}_{data}_s{cfg['seed']}" + (f"_{cfg['run_tag']}" if not _blank(cfg['run_tag']) else '')
    cfg['run_dir'] = os.path.join(cfg['out_root'], cfg['run_name'])
    if not os.path.isabs(cfg['owl_prompts']):
        cfg['owl_prompts'] = os.path.join(cfg['nad_dir'], cfg['owl_prompts'])
    return Config(cfg)


# ====================================================================== for_dataset 코드 가져오기 (읽기 전용)
def fd_paths(cfg):
    fd = cfg['for_dataset']
    return dict(build=os.path.join(fd, 'build'),
                seld=os.path.join(fd, 'baseline', 'seld_challenge'),
                ntu=os.path.join(fd, 'baseline', 'ntu_challenge'),
                main20=os.path.join(fd, 'dataset', 'main20'),
                s1=os.path.join(fd, 'generated', 's1'),
                starss=os.path.join(fd, 'STARSS23_datasets'))


def import_dsgen(cfg, raw_dir):
    """생성기 모듈(dsgen.py)을 그대로 import 한다. dsgen 은 import 할 때 환경 변수 RAW 를 읽는다."""
    os.environ['RAW'] = raw_dir
    p = fd_paths(cfg)['build']
    if p not in sys.path:
        sys.path.insert(0, p)
    import dsgen
    if dsgen.RAW != raw_dir:
        dsgen.RAW = raw_dir
    return dsgen


def add_seld_path(cfg):
    """베이스라인(seld_challenge)의 score.py, metrics.py, seld_label_utils.py, utils.py 를 쓰기 위해 경로를 더한다."""
    p = fd_paths(cfg)['seld']
    if p not in sys.path:
        sys.path.insert(0, p)


def import_ntu_utils(cfg):
    """NTU utils.py 를 'ntu_utils' 라는 이름으로 불러온다(베이스라인 utils 와 이름이 겹치지 않게)."""
    if 'ntu_utils' in sys.modules:
        return sys.modules['ntu_utils']
    path = os.path.join(fd_paths(cfg)['ntu'], 'utils.py')
    spec = importlib.util.spec_from_file_location('ntu_utils', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['ntu_utils'] = mod
    spec.loader.exec_module(mod)
    return mod


def read_split_info(cfg):
    import json
    return json.load(open(os.path.join(fd_paths(cfg)['main20'], 'split_info.json'), encoding='utf8'))
