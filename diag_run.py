"""
diag_run.py -- 1차 결과 진단 (DIGNOSIS_1.md). 추론만 한다.

    python -B diag_run.py <단계> --set drive_root="/content/drive/MyDrive/Colab Notebooks/ASR_2026-2" --set seed=1

단계
  prepare   val/test 오디오 캐시(로컬)와 정답 csv 만 준비 (train 캐시는 만들지 않음, 약 1.2GB)
  n2        1~4번: N2 후처리 4조합, 영상 원래/0/섞기, 앞/뒤 정확도, onscreen 점검  -> diag/DIAG_01/n2.json
  n1        3번(N1-otf 앞/뒤), 5번(N1-pre vs N1-otf 부트스트랩)                    -> diag/DIAG_01/n1.json
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
    a = ap.parse_args(argv)
    cfg = {p: load_config(p, a.set) for p in ('N1-pre', 'N1-otf', 'N2')}
    for c in cfg.values():
        add_seld_path(c)
    from nad import diag
    from nad.runner import Logger
    log = Logger(os.path.join(diag.out_dir(cfg['N2']), 'diag_out.log'))
    log(f'== 진단 {a.stage}: ' + ', '.join(c['run_name'] for c in cfg.values()))
    if a.stage == 'prepare':
        from nad.prerender import prepare_pre, extract_labels
        prepare_pre(cfg['N2'], ['val', 'test'], log)
        extract_labels(cfg['N2'], 'test', log)
    elif a.stage == 'n2':
        diag.run_n2(cfg['N2'], log)
    elif a.stage == 'n1':
        diag.run_n1(cfg['N1-pre'], cfg['N1-otf'], log)
    elif a.stage == 'tables':
        diag.write_tables(cfg['N2'], log)
    else:
        raise SystemExit(f'알 수 없는 단계: {a.stage}')


if __name__ == '__main__':
    main()
