# 🛰️ Data Communication: Distributed Inference Pipeline

이 프로젝트는 로컬 머신과 Raspberry Pi + Hailo-8 간의 분산 추론을 위한 통신 구조를 구현합니다.  
MobileNetV3 기반 CNN 특징 추출은 Hailo에서 수행하고, 이후 결과는 로컬에서 GRU+MLP를 거쳐 도로 상태를 예측합니다.

---

## 📁 디렉토리 구조

<pre lang="markdown"> ## 📁 디렉토리 구조 
``` data_communication/ ├── Client/ │ ├── requirements.txt # 클라이언트 환경 설치 파일 │ ├── second_cnn_feature_extractor.hef # CNN 모델 (Hailo용) │ └── server2pi2server.py # 로컬 → 라즈베리파이 통신 클라이언트 │ ├── Server/ │ ├── GRU_MLP_xpu.py # GRU+MLP 분류기 정의 │ ├── Inference.py # 전체 추론 파이프라인 │ ├── best_gru_mlp_classifier.pth # 학습된 GRU+MLP 모델 가중치 │ └── __pycache__/ # 파이썬 캐시 파일 │ ├── normal_road.mp4 # 테스트 영상 파일 └── requirements.txt # 서버 환경 설치 파일 ``` </pre>

---

## 🚀 실행 순서

1. Raspberry Pi + Hailo-8 측에서 추론 서버 구동
2. 로컬 클라이언트에서 영상 프레임 단위 전송
3. Hailo-8에서 CNN 추론 후 특징 벡터 반환
4. 로컬 서버에서 GRU + MLP로 도로 상태 분류
5. 결과를 영상 위에 시각화 (OpenCV)

---

