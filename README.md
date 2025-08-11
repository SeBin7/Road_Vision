## Road_Vision

* 자율주행 환경에서 노면 상태는 주행 안정성과 안전에 큰 영향을 미칩니다. 본 연구에서는 파손 도로, 정상 도로, 젖은 도로, 결빙 도로를 실시간으로 탐지하고 분류하는 딥러닝 기반 시스템을 제안합니다.

* 본 시스템은 Raspberry Pi에 탑재된 Hailo-8 AI 가속기를 통해 경량 CNN 기반 특징 추출을 엣지에서 수행합니다. 이때 추출된 피처 벡터는 서버로 전송되어 GRU 기반 시계열 모델을 통해 최종 도로 상태를 분류합니다.
서버는 CARLA 시뮬레이터와 연동되어 분류 결과를 실시간으로 시각화하며, 엣지-서버 분산 구조를 통해 연산 효율성과 실시간성을 모두 확보하였습니다.


## System Flow Chart

<img width="1143" height="439" alt="image" src="https://github.com/user-attachments/assets/b60cdc1d-6c16-4a39-ad7a-9979a04f8337" />



## Sequence Diagram

<img width="987" height="618" alt="image" src="https://github.com/user-attachments/assets/22fef8bb-28f6-40c8-9458-ec0bc24ec377" />

## Deeplearing Flow Chart

<img width="504" height="780" alt="image" src="https://github.com/user-attachments/assets/74d33f0b-9fbc-4bdf-a6cf-4d2a5a638095" />


## Output


<img width="1653" height="1002" alt="image" src="https://github.com/user-attachments/assets/cf7c82c0-7f98-411e-acf9-59f45451d6df" />
<img width="1371" height="862" alt="image" src="https://github.com/user-attachments/assets/1f1b2190-393e-4804-95ed-3b60e9238428" />
