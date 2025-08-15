# 🚗 Road_Vision

자율주행 환경에서 노면 상태(파손/정상/젖은/결빙)를 **실시간**으로 탐지·분류하는 딥러닝 시스템입니다.  
엣지(Raspberry Pi + Hailo-8)에서 **경량 CNN**으로 피처를 추출하고, 서버에서 **GRU 기반 시계열 분류**를 수행합니다.  
CARLA 시뮬레이터와 연동하여 결과를 시각화하고, 엣지–서버 분산 구조로 실시간성과 효율성을 확보했습니다.

---

## 🧭 System Flow

- System Flow Chart  
  <img width="1681" height="779" alt="System Flow" src="https://github.com/user-attachments/assets/8d09a11b-3b33-4c25-aeb3-4721b45c68a8" />



- Sequence Diagram  
  <img width="987" height="618" alt="Sequence Diagram" src="https://github.com/user-attachments/assets/22fef8bb-28f6-40c8-9458-ec0bc24ec377" />



- Deep Learning Flow    
  <img width="1888" height="1133" alt="Deep Learning Flow" src="https://github.com/user-attachments/assets/7f474819-e9c4-4380-b2dc-ce8909b982d3" />


---

## 📂 Project Structure

```plaintext
ROAD_VISION/
├── data/                        # (로컬) 데이터셋/샘플
├── demo/                        # 데모 스크립트/예제
├── doc/                         # 문서/다이어그램/보고서
├── models/                      # 체크포인트(.pth/.onnx 등)
├── src/
│   ├── calibration/             # Hailo-8/모델 캘리브레이션
│   ├── datasets/                # 데이터셋/전처리 모듈
│   ├── inference/               # 추론 엔진/시각화
│   ├── integration/             # CARLA/데스크톱 통신 등 통합 코드
│   ├── models/                  # CNN/GRU/MLP 등 모델 정의
│   ├── tools/                   # 변환/util/비디오 처리
│   ├── trainer/                 # 학습 루프/평가/메트릭
│   └── vis/                     # Grad-CAM 등 시각화
├── config.yaml                  # 공통 설정 (경로/하이퍼파라미터)
├── requirements.txt             # 기본 의존성
├── requirements_xpu.txt         # Intel XPU 환경용(옵션)
└── README.md
```

---

## ⚙️ Quick Start

### 1) 환경 세팅 (권장: 가상환경)
```bash
python3 -m venv .venv
source .venv/bin/activate

# 기본 CPU/GPU 환경
pip install -r requirements.txt

# (옵션) Intel XPU 환경 — requirements_xpu.txt 사용
# pip install -r requirements_xpu.txt
```

### 2) 설정 파일
config.yaml에서 경로 및 하이퍼파라미터를 조정합니다.
```bash
device: "cpu"            # "cpu" | "cuda" | "xpu"
epochs: 20
batch_size: 16
learning_rate: 0.001
seq_len: 10
resize: [224, 224]
train_data_dir: "./data/train"
val_data_dir: "./data/val"
cnn_weight: "./models/best_cnn.pth"
cls_weight: "./models/best_gru.pth"

## 🏋️ Training (예시)

아래 스크립트와 모듈명은 실제 사용 파일에 맞게 변경 가능합니다.

```bash
# 4채널 학습
python src/trainer/Train_4ch_improved_validation.py \
  --config config.yaml

# 5채널(npz) 학습
python src/trainer/Train_5ch_npz.py \
  --config config.yaml
```

## 🔎 Inference (예시)

```bash
# 4채널 추론
python src/inference/infer_4ch.py \
  --config config.yaml --video ./data/sample.mp4

# 5채널(npz) 추론
python src/inference/Infer_ch5_npz.py \
  --config config.yaml --npy ./data/sample.npy
```

## 🧩 Integration / Demo

### 1. CARLA 연동 추론
```bash
python src/integration/carla/carla_comm/4ch_carla/infer_4ch.py \
  --config config.yaml
```

### 2. 🖥 Desktop 통신 모드

**서버 실행**
```bash
python src/integration/desktop_comm/server/app.py
```

**클라이언트 실행**
```bash
python src/integration/desktop_comm/client/app.py \
  --server http://127.0.0.1:5000
```

**특징**
- 로컬 또는 원격 서버 주소 지정 가능
- 실시간 영상 스트리밍 + GRU 분류 결과 수신 및 시각화
- 배치 크기와 시퀀스 길이 조정 가능

---

## 3. 📸 Output (샘플)

<img width="1653" height="1002" alt="output-1" src="https://github.com/user-attachments/assets/cf7c82c0-7f98-411e-acf9-59f45451d6df" />
<img width="1371" height="862" alt="output-2" src="https://github.com/user-attachments/assets/1f1b2190-393e-4804-95ed-3b60e9238428" />
