"""
nad_run.py -- 새 모델(NAD)의 모든 단계를 이 파일 하나로 돌린다.

    python -B nad_run.py <단계> --preset N1-pre --set drive_root="/content/drive/MyDrive/Colab Notebooks/ASR_2026-2" --set seed=1

단계
  prepare_raw   STARSS23 원본(FOA, 라벨)에서 train 방 녹음만 로컬에 푼다 (즉석 변환, 녹음 단위 CLAP 용)
  check_match   [필수 확인] 즉석 변환 결과가 생성기 출력(main20 train 클립)과 같은지
  pre_size      미리 렌더링 오디오 캐시 예상 용량 (만들기 전에 먼저 본다)
  prepare_pre   미리 렌더링한 데이터의 오디오 캐시(로컬)와 라벨 캐시(드라이브), val 정답 풀기
  check_feat    GPU 음향 특징이 NTU numpy 특징과 같은지 (val 클립 1개)
  oracle        정답 -> 새 출력 형식 -> 후처리 -> csv -> 베이스라인 채점이 정상인지
  w1_check      [W1, --preset N2w] 소프트 onscreen 캐시 + 일치율, θ 복원, α 형식 오라클, 정답 입력 손실 (MODEL_DEV_2.md 확인 2, 3)
  clap          CLAP 캐시 (train 녹음 단위 + val/test 클립 단위)
  owl_size      OWL-ViT 캐시 예상 용량 (만들기 전에 먼저 본다)
  owl_check     OWL-ViT 전처리 설정 확인 (정사각형 늘리기/여백, 열 방위각)
  owl           OWL-ViT 캐시 (PCA 학습 + train/반전/val/test)
  train | eval | tune_post | predict | score_test
설정은 configs/nad_base.yaml, 프리셋은 nad/config.py 의 PRESETS.
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
    ap.add_argument('--preset', default=None)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    a = ap.parse_args(argv)
    cfg = load_config(a.preset, a.set)
    add_seld_path(cfg)
    print(f'실행 이름: {cfg["run_name"]} | 단계: {a.stage} | 결과 폴더: {cfg["run_dir"]}', flush=True)
    s = a.stage
    dev = 'cuda' if _cuda() else 'cpu'
    all_train = ['train_s0', 'train_s1']
    if s == 'prepare_raw':
        from nad.onthefly import prepare_raw
        prepare_raw(cfg)
    elif s == 'check_match':
        from nad.onthefly import check_match
        r = check_match(cfg, n_clips=3)
        print('통과' if all(x['PASS'] for x in r) else '불일치 있음')
    elif s == 'pre_size':
        from nad.prerender import size_report
        size_report(cfg, all_train + ['val', 'test'])
    elif s == 'prepare_pre':
        from nad.prerender import prepare_pre
        prepare_pre(cfg, all_train + ['val', 'test'])
    elif s == 'check_feat':
        import numpy as np
        from nad.prerender import load_audio
        from nad.features import check_against_ntu
        names, arr = load_audio(cfg, 'val')
        check_against_ntu(cfg, arr[5].astype(np.float32) / 32768.0, device=dev)
    elif s == 'oracle':
        from nad.runner import oracle_check
        oracle_check(cfg)
    elif s == 'w1_check':
        from nad.w1check import run
        run(cfg)
    elif s == 'clap':
        from nad.semantic import build_rec_train, build_clips
        build_rec_train(cfg, dev)
        for k in ('val', 'test'):
            build_clips(cfg, k, dev)
    elif s == 'owl_size':
        from nad.visual import size_report
        size_report(cfg, all_train + ['val', 'test'])
    elif s == 'owl_check':
        from nad.visual import OwlEncoder, _read_frames
        from nad.prerender import list_members
        mem = list_members(cfg, 'val'); n = sorted(mem)[0]
        os.makedirs(os.path.join(cfg['local_root'], 'tmp'), exist_ok=True)
        fr = _read_frames(mem[n][0], mem[n][1]['video'], os.path.join(cfg['local_root'], 'tmp'))
        OwlEncoder(cfg, dev).check(fr[100])
    elif s == 'owl':
        from nad.visual import build
        build(cfg, [{0: 'train_s0', 1: 'train_s1'}[x] for x in cfg['train_stages']] + ['val', 'test'], dev)
    elif s == 'train':
        from nad.runner import stage_train
        stage_train(cfg)
    elif s == 'eval':
        from nad.runner import stage_eval
        stage_eval(cfg)
    elif s == 'tune_post':
        from nad.runner import stage_tune_post
        stage_tune_post(cfg)
    elif s == 'predict':
        from nad.runner import stage_predict
        stage_predict(cfg)
    elif s == 'score_test':
        from nad.runner import stage_score_test
        stage_score_test(cfg)
    else:
        raise SystemExit(f'알 수 없는 단계: {s}')


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == '__main__':
    main()
