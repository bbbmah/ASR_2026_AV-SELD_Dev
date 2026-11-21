# ASR_2026_AV-SELD_Dev

Moving-Camera AV-SELD 새 모델(NAD) 개발 코드와 문서.

- 실행: `colab_nad.ipynb`(학습·평가), `colab_diag.ipynb`(1차 결과 진단, 추론만)
- 명령: `python -B nad_run.py <단계> --preset ... --set key=value`, `python -B diag_run.py <단계> --set ...`
- 파일 구성과 구현 중 정한 것: `NOTES.md`
- 설계 문서: `MODEL_DEV_1.md`, `MODEL_DEV_2.md` / 진단: `DIGNOSIS_1.md`, `diag/DIAG_01/DIAG_01.md`

데이터셋, 캐시, 체크포인트(`cache/`, `runs/`)와 베이스라인·생성기 코드(`for_dataset`)는 이 저장소에 포함하지 않는다.
