## Road_Vision

* 차량 탑재 YOLO와 RNN 기반의 노면이상탐지시스템을 통해 결빙, 젖은 노면, crack 등의 실시간 감지 및 V2X 통신 기반 차량 간 정보공유를 통한 공공데이터 구축을 최종 목표로 설정할 수 있습니다

## Use Case

<img width="1684" height="1204" alt="image" src="https://github.com/user-attachments/assets/3a95f96b-71ab-491b-8a85-ce132c013793" />


## High Level Design

<img width="2724" height="1284" alt="image" src="https://github.com/user-attachments/assets/e34c750a-b606-4502-b9fe-e5641502c8d6" />


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
