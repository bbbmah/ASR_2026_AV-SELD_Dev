"""
err_run.py -- 오라클 교체와 오류 분해 (nad/errdiag.py). 기존 예측 csv 만 읽는다. GPU 필요 없음.

    python -B err_run.py run    --set drive_root="/content/drive/MyDrive/Colab Notebooks/ASR_2026-2" --set seed=1
    python -B err_run.py tables --set drive_root=...

  run     --runs 의 각 run, --splits 의 각 split 에 대해 계산 -> diag/ERR_01/<run>_<split>[_last].json
  tables  폴더의 json 을 모두 모아 표 -> diag/ERR_01/tables.md
"""
import sys
sys.dont_write_bytecode = True
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nad.config import load_config

DEF_RUNS = 'nad_N2_S01_s1,nad_N2w_S01_s1,nad_N2w-a5_S01_s1,nad_N1w-a5_otf_s1'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['run', 'tables'])
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    ap.add_argument('--runs', default=DEF_RUNS, help='쉼표로 구분한 run 이름 (runs/<이름>/pred_<split>)')
    ap.add_argument('--splits', default='val,test')
    ap.add_argument('--last', action='store_true', help='pred_<split>_last 도 계산')
    a = ap.parse_args(argv)
    from nad import errdiag
    from nad.runner import Logger
    import yaml
    base = load_config('N2', a.set)
    log = Logger(os.path.join(errdiag.out_dir(base), 'err_out.log'))
    if a.stage == 'tables':
        errdiag.write_tables(base, log)
        return
    runs = [r for r in a.runs.split(',') if r]
    log(f'== 오류 분해: {runs}, split {a.splits}, last {a.last}')
    for run in runs:
        y = os.path.join(base['out_root'], run, 'config_used.yaml')
        preset = yaml.safe_load(open(y, encoding='utf8'))['preset']
        cfg = load_config(preset, a.set)          # 경로와 modality(1단계 audio, 2단계 audio_visual)만 쓴다
        errdiag.run_all(cfg, [run], a.splits.split(','), a.last, log)
    errdiag.write_tables(base, log)


if __name__ == '__main__':
    main()
