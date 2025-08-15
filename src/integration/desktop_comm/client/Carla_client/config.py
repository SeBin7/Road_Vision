# config.py

import pygame
import carla

# 화면/시뮬레이션 파라미터
WIDTH = 800
HEIGHT = 600
FULLSCREEN_WIDTH = 1920
FULLSCREEN_HEIGHT = 1080
FPS = 30
SEQ_LEN = 10
FEATURE_DIM = 128
NUM_CLASSES = 2

# 모델/서버/엔드포인트 정보
MODEL_PATH = './gru_mlp_classifier_3ch_carla.pth'
SERVER_URL = 'http://192.168.48.123:6000/infer'
EDGE_URL = "http://192.168.48.123:5000"

# CARLA 연결 정보
CARLA_HOST = '192.168.48.120'
CARLA_PORT = 2000
WORLD_NAME = 'Town03'

# 레이블 매핑
LABEL_MAP = {0: 'normal_road', 1: 'wet_road'}

# feature vector 크기
FEATURE_DIM_SERVER = 32
FEATURE_DIM_MODEL = 128

# 날씨 프리셋 (키: pygame 키코드, 값: carla.WeatherParameters)
weather_presets = {
    pygame.K_1: carla.WeatherParameters.ClearNoon,
    pygame.K_2: carla.WeatherParameters.CloudyNoon,
    pygame.K_3: carla.WeatherParameters.WetNoon,
    pygame.K_4: carla.WeatherParameters.MidRainyNoon,
    pygame.K_5: carla.WeatherParameters.WetCloudyNoon,
    pygame.K_6: carla.WeatherParameters.HardRainNoon,
    pygame.K_7: carla.WeatherParameters.ClearNight,
    pygame.K_8: carla.WeatherParameters.HardRainNight
}
