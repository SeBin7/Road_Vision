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


## Clone code


```shell
git clone https://github.com/SeBin7/Road_Vision.git
```

## Prerequite

* (프로잭트를 실행하기 위해 필요한 dependencies 및 configuration들이 있다면, 설치 및 설정 방법에 대해 기술)

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Steps to build

* (프로젝트를 실행을 위해 빌드 절차 기술)

```shell
cd ~/xxxx
source .venv/bin/activate

make
make install
```

## Steps to run

* (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

```shell
cd ~/xxxx
source .venv/bin/activate

cd /path/to/repo/xxx/
python demo.py -i xxx -m yyy -d zzz
```

## Output

* (프로젝트 실행 화면 캡쳐)

![./result.jpg](./result.jpg)

## Appendix

* (참고 자료 및 알아두어야할 사항들 기술)
