"""
diag_run.py -- 1차 결과 진단 (DIGNOSIS_1.md). 추론만 한다.

    python -B diag_run.py <단계> --set drive_root="/content/drive/MyDrive/Colab Notebooks/ASR_2026-2" --set seed=1
    W1 (MODEL_DEV_2.md 보고 2, 3번):  python -B diag_run.py n2 --n2 N2w --diag W1_N2w --combo a --pp_from nad_N2_S01_s1 --set ...
    W1 비교 부트스트랩:  python -B diag_run.py n1 --n2 N2w --diag W1_N2w --n1a N1-otf --n1b N1w                       -> n1.json
                         python -B diag_run.py n1 --n2 N2w --diag W1_N2w --n1a N2 --n1b N2w --n1_out n1_N2 --pp_from nad_N2_S01_s1

단계
  prepare   val/test 오디오 캐시(로컬)와 정답 csv 만 준비 (train 캐시는 만들지 않음, 약 1.2GB)
  n2        1~4번: N2 후처리 4조합, 영상 원래/0/섞기, 앞/뒤 정확도, onscreen 점검  -> diag/DIAG_01/n2.json
  n1        3번(--n1b 앞/뒤), 5번(--n1a vs --n1b 부트스트랩, 기본 N1-pre vs N1-otf) -> diag/<이름>/<--n1_out>.json
  tables    두 json 으로 표를 만든다                                               -> diag/DIAG_01/tables.md
"""
import sys
sys.dont_write_bytecode = True
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nad.config import load_config, add_seld_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    ap.add_argument('--n2', default='N2', help='n2 단계에서 진단할 2단계 프리셋 (W1: N2w)')
    ap.add_argument('--diag', default='DIAG_01', help='결과 폴더 diag/<이름> (W1: W1_N2w)')
    ap.add_argument('--combo', default=None, help='후처리 조합 고정(a~d). 비우면 val 최고 조합')
    ap.add_argument('--pp_from', default=None, help='--n2 프리셋에만 적용할 postproc_from (W1: nad_N2_S01_s1)')
    ap.add_argument('--n1a', default='N1-pre', help='n1 단계 비교 A (기존 예측 csv)')
    ap.add_argument('--n1b', default='N1-otf', help='n1 단계 비교 B (앞/뒤 표는 B 로 만든다)')
    ap.add_argument('--n1_out', default='n1', help='n1 단계 결과 json 이름 (기본 n1 -> n1.json)')
    a = ap.parse_args(argv)
    load = lambda p: load_config(p, a.set + ([f'postproc_from={a.pp_from}'] if (a.pp_from and p == a.n2) else []))
    cfg = {'N1-pre': load(a.n1a), 'N1-otf': load(a.n1b), 'N2': load(a.n2)}
    for c in cfg.values():
        add_seld_path(c)
    from nad import diag
    diag.DIAG_ID = a.diag
    from nad.runner import Logger
    log = Logger(os.path.join(diag.out_dir(cfg['N2']), 'diag_out.log'))
    log(f'== 진단 {a.stage}: ' + ', '.join(c['run_name'] for c in cfg.values()))
    if a.stage == 'prepare':
        from nad.prerender import prepare_pre, extract_labels
        prepare_pre(cfg['N2'], ['val', 'test'], log)
        extract_labels(cfg['N2'], 'test', log)
    elif a.stage == 'n2':
        diag.run_n2(cfg['N2'], log, a.combo)
    elif a.stage == 'n1':
        diag.run_n1(cfg['N1-pre'], cfg['N1-otf'], log, a.n1_out)
    elif a.stage == 'tables':
        diag.write_tables(cfg['N2'], log)
    else:
        raise SystemExit(f'알 수 없는 단계: {a.stage}')


if __name__ == '__main__':
    main()
