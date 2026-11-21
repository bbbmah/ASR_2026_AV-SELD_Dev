# NOTES — 새 모델(NAD) 구현 기록

## 파일
| 파일 | 내용 |
|---|---|
| `colab_nad.ipynb` | 코랩 실행 노트북 (위에서부터 순서대로) |
| `nad_run.py` | 모든 단계 명령 (`python -B nad_run.py <단계> --preset ... --set key=value`) |
| `configs/nad_base.yaml`, `configs/owl_prompts.yaml` | 설정 기본값, OWL-ViT 클래스 문장 |
| `nad/onthefly.py` | 즉석 변환 로더 (생성기 `dsgen.py` 함수 그대로 import) |
| `nad/prerender.py` | 미리 렌더링 데이터 캐시 (zip 에서 wav/라벨만 읽음) |
| `nad/features.py` | (A) 음향 특징 7채널 GPU 계산 (NTU 식 + ΔIV) |
| `nad/semantic.py`, `nad/visual.py` | (B) CLAP 캐시, (C) OWL-ViT 열 특징 캐시 |
| `nad/model.py`, `nad/loss.py`, `nad/postproc.py` | 모델 (A~F), 새 ADPIT 손실, 추론 후처리 |
| `nad/data.py`, `nad/runner.py` | 데이터셋, 학습(이어 하기)/평가/예측/채점 |
| `colab_diag.ipynb`, `diag_run.py`, `nad/diag.py` | 1차 결과 진단(DIGNOSIS_1.md, 추론만) -> `diag/DIAG_01/` |

프리셋: `N1-pre`(1단계, main20 train+S1), `N1-otf`(1단계, 즉석 변환), `N2`(2단계 AV, `init_from`=1단계 실행 이름).

## 정한 것 (CLAUDE.md 에 없거나 애매했던 부분)
- 1단계는 앞/뒤, onscreen 손실이 꺼져 있으므로 추론 때 모두 앞(+), onscreen 0 으로 낸다. 모델 선택 기준은 1단계 val `F_basic`, 2단계 val `F_ext`.
  (생성기 스테레오는 W±(sinθ·X+cosθ·Y) 이라 앞/뒤 대칭이다. 소리만으로는 앞/뒤를 알 수 없다.)
- CLAP: train 은 원본 녹음 W(=Mid, 궤적·반전과 무관) 에서 녹음 단위로 0.5초 간격 창을 뽑고, 클립 시작 프레임으로 잘라 보간한다(N1-pre, N1-otf 공통).
  val/test 는 클립 Mid 에서 클립 단위. 클립마다 뽑으면 train 만 약 33만 창이라 바꿨다. 클립 가장자리 1초만 창이 클립 밖까지 걸치는 점이 다르다.
- 2단계는 미리 렌더링 데이터로만 학습한다(즉석 변환 영상에 OWL-ViT 를 매번 돌리는 것은 비용이 커서). 1단계에서 즉석 변환이 이기면 그 1단계 모델을 `init_from` 으로 쓴다((A) 는 고정).
- 즉석 변환 녹음 길이는 오디오 길이로 잰다(생성기는 오디오·영상 중 짧은 쪽, 차이 최대 0.3초). 영상 없는 녹음(room21 mix001~012)은 기본 제외(`otf_include_no_video`).
- 좌우 반전 영상 캐시는 거울이 아닌 train 클립만 만든다(S1 거울 클립은 다시 뒤집지 않음).
- 특징 정규화 통계는 학습 데이터 처음 40배치로 구해 실행 폴더에 저장한다(N2 는 1단계 통계를 그대로 씀).
- 학습: AdamW, 학습률 1e-3, 워밍업 1000, 코사인(최저 5%), 배치 32, 총 20,000 갱신, 에폭 = 500 갱신. 모두 제안값.

## 확인한 것 (로컬 CPU, 2026-11-20)
- 즉석 변환 = 생성기 출력: main20 train 3클립(train_00311, train_01025, train_01587)에서 오디오 비트 단위 일치, 라벨·궤적 일치.
  train 방 녹음 59개(영상 있는 것), 3.08시간, FOA 메모리 1.99GB. 생성기에는 게인 정규화가 없다.
- GPU 특징 식 = NTU numpy 식 (val_00005): log-mel 최대 차이 약 2e-4 dB, IV 5e-7, MSC 2e-5 (float32 오차 수준).
- 정답 -> 새 출력 형식 -> 후처리 -> csv -> 베이스라인 채점: val 123클립 F_ext 99.3%, DOA 0.05°.
- 작은 설정(배치 2, 갱신 몇 번, 가짜 CLAP/OWL 캐시)으로 N1-otf 학습 -> 이어 하기 -> eval, N2 학습(1단계 불러오기, 반전 영상, 중간 고정 해제) -> tune_post -> eval 이 끝까지 돈다.
- 확인 못 한 것: 실제 CLAP/OWL-ViT 실행(로컬에 transformers 없음), GPU 속도와 메모리, 전체 학습. OWL-ViT 전처리 방식은 `owl_check` 가 코랩에서 processor 설정을 읽어 출력한다.

## 사전학습 가중치 (출처)
- LAION-CLAP: `laion/clap-htsat-unfused` (HuggingFace)
- OWL-ViT: `google/owlvit-base-patch32` (HuggingFace)
