# Road-Vision C++ 추론 (OpenCV + LibTorch, XPU/CPU)

이 폴더는 Road-Vision의 영상 추론 파이프라인을 **C++로 구현**한 런너입니다.  
목표는 “모델 커널을 새로 최적화”하는 것이 아니라, **프레임 처리 파이프라인(전처리/전송/스케줄링/메모리)**을 최적화해서 FPS를 끌어올리는 것입니다.

---

## 산출물(모델 파일)

루트 기준 아래 경로를 사용합니다.
- `models/torchscript/road_vision_ts.pt` (XPU 실행용 TorchScript)
- `models/torchscript/road_vision_ts_cpu.pt` (CPU 실행용 TorchScript)

> 주의: XPU로 export된 TorchScript는 내부에 `xpu:0` 디바이스가 bake되어 CPU에서 실패할 수 있습니다.  
> CPU 비교는 `road_vision_ts_cpu.pt`를 사용하세요.

---

## 빌드

권장(스크립트):
```bash
source ../.xpu/bin/activate
./build_cpp_xpu.sh
```

직접(CMake):
```bash
cmake -S . -B build
cmake --build build -j
```

---

## 실행

공통:
```bash
source ../.xpu/bin/activate
./run_cpp_xpu.sh <model.pt> <video.mp4>
```

CPU 강제(안정 비교):
```bash
RV_FORCE_CPU=1 ./run_cpp_xpu.sh ../models/torchscript/road_vision_ts_cpu.pt ../data/normal_road.mp4
```

Hybrid 권장(안정/성능): CNN=XPU, CLS=CPU
```bash
RV_CNN_XPU_CLS_CPU=1 RV_CPU_FALLBACK_MODEL=../models/torchscript/road_vision_ts_cpu.pt \
  ./run_cpp_xpu.sh ../models/torchscript/road_vision_ts.pt ../data/normal_road.mp4
```

XPU 단독(환경에 따라 강제종료가 발생할 수 있음):
```bash
./run_cpp_xpu.sh ../models/torchscript/road_vision_ts.pt ../data/normal_road.mp4
```

---

## 로그 포맷(비교용)

실행 시 다음 로그를 출력합니다.
- `[stage] read=cpu preproc=cpu infer=... ui=cpu`
- `[timing] read=...ms preproc=...ms infer=...(cpu|xpu|hybrid)ms ui=...ms total=...ms fps≈...`

---

## 환경변수

### 디바이스/안정성
- `RV_FORCE_CPU=1` 또는 `RV_DISABLE_XPU=1`: CPU 강제
- `RV_CNN_XPU_CLS_CPU=1`: Hybrid 모드(CNN=XPU, CLS=CPU)
- `RV_CPU_FALLBACK=1`: XPU 실행 중 예외가 throw로 떨어지면 CPU 모델로 fallback 시도
- `RV_CPU_FALLBACK_MODEL=/abs/or/rel/path/to/road_vision_ts_cpu.pt`: CPU fallback/Hybrid에서 사용할 CPU TorchScript 경로

### 디버그
- `RV_SPLIT_DEBUG=1`: 모델의 `cnn/cls` submodule을 분리 실행(마커 로그)
- `RV_DEBUG_TRACE=1` 또는 `RV_DEBUG_FRAMES=N`: 초기 N프레임 동안 단계별 마커 로그

### (고급) XPU 런타임 실험 옵션
- `RV_XPU_EXTERNAL_QUEUE=1`: 외부 SYCL queue를 설치(환경에 따라 요구사항/효과가 다름)
- `RV_XPU_KEEP_ALIVE=1`: 결과 텐서를 잠시 유지(디버깅)
- `RV_XPU_EMPTY_CACHE=1`: 프레임마다 XPU 캐시 비우기(성능 저하 가능)

---

## C++ 최적화 포인트(상세)

### 1) 파이프라이닝(스레딩)
`read+preproc(CPU)`와 `infer(XPU/CPU)`를 Producer–Consumer 구조로 겹쳐 실행합니다.  
구현: `src/rv/pipeline.cpp`

### 2) 전송량/정규화 경로 최적화
전처리 결과를 `(1,4,224,224) uint8`로 만들어 H2D 전송량을 줄이고, 정규화는 디바이스에서 in-place로 수행합니다.  
구현: `src/rv/preprocess.cpp`, `src/rv/xpu_engine.cpp`

### 3) in-place + 버퍼 재사용
매 프레임 `torch::empty`를 만들지 않고 입력 버퍼(`input_buf_`)를 재사용하며, `copy_ / sub_ / div_`로 중간 텐서 생성을 줄입니다.  
구현: `src/rv/xpu_engine.cpp`, `src/rv/hybrid_engine.cpp`

### 4) Hybrid 모드(현실적 안정/성능)
XPU에서 불안정할 수 있는 GRU/분류기(또는 디바이스 혼합 문제)를 CPU로 우회하면서도, CNN 가속 이득을 유지합니다.  
구현: `src/rv/hybrid_engine.cpp`

---

## 코드 구조(리팩토링 결과)

핵심은 “엔트리포인트”와 “기능”을 분리한 것입니다.
- 엔트리/CLI: `src/road_vision_xpu.cpp`
- 파이프라인: `src/rv/pipeline.cpp`
- 전처리: `src/rv/preprocess.cpp`
- 엔진(Strategy/Factory):
  - 인터페이스/팩토리: `src/rv/engine.h`, `src/rv/engine_factory.cpp`
  - CPU: `src/rv/cpu_engine.cpp`
  - XPU: `src/rv/xpu_engine.cpp`
  - Hybrid: `src/rv/hybrid_engine.cpp`
- 오버레이/UI: `src/rv/overlay.cpp`
