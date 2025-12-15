## Road_Vision

자율주행 환경에서 노면 상태(파손/정상/젖은/결빙)를 **실시간**으로 탐지·분류하는 딥러닝 시스템입니다.  
엣지(Raspberry Pi + Hailo-8)에서 **경량 CNN**으로 피처를 추출하고, 서버에서 **GRU 기반 시계열 분류**를 수행합니다.  
CARLA 시뮬레이터와 연동하여 결과를 시각화하고, 엣지–서버 분산 구조로 실시간성과 효율성을 확보했습니다.


이 레포는 다음 실행 경로를 제공합니다.
- 로컬 추론(Python): `src/inference/`
- 분산 추론(Hailo-8 + 로컬 GRU): `src/integration/desktop_comm/`
- CARLA 연동: `src/integration/carla/`, `src/integration/desktop_comm/client/Carla_client/`
- C++/LibTorch 추론(최적화): `cpp/`

---

## System Flow

- System Flow Chart  
  <img width="1681" height="779" alt="System Flow" src="https://github.com/user-attachments/assets/8d09a11b-3b33-4c25-aeb3-4721b45c68a8" />

- Sequence Diagram  
  <img width="987" height="618" alt="Sequence Diagram" src="https://github.com/user-attachments/assets/22fef8bb-28f6-40c8-9458-ec0bc24ec377" />

- Deep Learning Flow    
  <img width="1888" height="1133" alt="Deep Learning Flow" src="https://github.com/user-attachments/assets/7f474819-e9c4-4380-b2dc-ce8909b982d3" />

---

## Project Structure

```plaintext
ROAD_VISION/
├── cpp/                         # C++/LibTorch 추론(빌드/실행 스크립트 포함)
├── data/                        # (로컬) 데이터셋/샘플
├── demo/                        # 데모 스크립트/예제
├── doc/                         # 문서/다이어그램/보고서
├── models/                      # 체크포인트(.pth/.onnx 등)
│   ├── torchscript/             # C++ 추론용 TorchScript(.pt) 산출물
├── src/
│   ├── inference/               # Python 추론/시각화
│   ├── integration/             # CARLA/데스크톱 통신(분산 추론) 등 통합 코드
│   ├── models/                  # CNN/GRU/MLP 등 모델 정의
│   └── trainer/                 # 학습/평가
├── requirements.txt
├── requirements_xpu.txt
└── README.md
```

---

## 로컬 추론(Python)

### Python (Live + Lite)
```bash
python src/inference/infer_batch_xpu_preproc_fast.py data/normal_road.mp4 --live --live-lite
```

타이밍 로그(C++ 포맷에 맞춤):
```bash
ROAD_VISION_TIMING=1 ROAD_VISION_TIMING_INT=30 \
  python src/inference/infer_batch_xpu_preproc_fast.py data/normal_road.mp4 --live --live-lite
```

CPU 강제:
```bash
ROAD_VISION_CPU=1 python src/inference/infer_batch_xpu_preproc_fast.py data/normal_road.mp4 --live --live-lite
```

---

## 분산 추론(Hailo-8 + 로컬 GRU)

Raspberry Pi + Hailo-8에서 CNN 특징 추출을 수행하고, 로컬에서 GRU+MLP로 분류하는 구조입니다.  
자세한 실행/구성은 `src/integration/desktop_comm/README.md`를 참고하세요.

---

## CARLA 연동

CARLA 시뮬레이터 기반의 통합/데모 코드가 포함되어 있습니다.
- CARLA 클라이언트 문서: `src/integration/desktop_comm/client/Carla_client/README.md`
- CARLA 학습/추론 관련 코드: `src/integration/carla/`

---

## C++ 최적화

C++/LibTorch 파이프라인을 추가하여, 동일 입력(video) 기준으로 FPS가 개선되었습니다.

- 측정 조건(요약): `data/normal_road.mp4`, Lite UI, 30프레임 평균, 첫 구간(warmup) 제외
- 측정 예시(로그에서 발췌, ms는 30프레임 평균)
  - Python(XPU): total `13.82~19.47ms` → `fps≈51.4~72.4`
    - read `0.47~0.59ms`, preproc `3.39~5.37ms`, infer `6.44~9.54ms`, ui `3.52~5.15ms`
    - warmup(첫 30프레임): total `49.21ms` → `fps≈20.3`
  - C++ Hybrid(CNN=XPU, CLS=CPU): total `10.30~10.65ms` → `fps≈93.9~97.1`
    - read `0.41~0.49ms`, preproc `0.72~0.81ms`, infer `4.84~5.02ms`, ui `4.28~4.38ms`
    - warmup(첫 30프레임): total `43.17ms` → `fps≈23.2`
  - 개선 배율(대략): `~1.3x ~ 1.8x` (구간/환경에 따라 변동)

C++ 빌드/실행/환경변수/최적화 상세는 `cpp/README.md` 참고.

---

## 학습/통합

- 학습: `src/trainer/` 아래 스크립트 사용 (실험 설정은 상황에 맞게 조정)
- CARLA/통합: `src/integration/` 아래 모듈 사용

---

## Output

<img width="1260" height="780" alt="output-1" src="https://github.com/user-attachments/assets/cf7c82c0-7f98-411e-acf9-59f45451d6df" />
<img width="1260" height="780" alt="output-2" src="https://github.com/user-attachments/assets/1f1b2190-393e-4804-95ed-3b60e9238428" />
<img width="1260" height="780" alt="output-1" src="https://github.com/user-attachments/assets/da299684-4d95-42ec-86ac-e8cb600ddddc" />
<img width="1260" height="780" alt="output-1" src="https://github.com/user-attachments/assets/5f71adb6-39ca-4f51-9684-898632d4495b" />

