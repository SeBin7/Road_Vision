# camera_manager.py
import numpy as np
from PIL import Image
import pygame
from queue import Full
from utils import encode_frame_to_base64

# 화면 출력용 surface 변수 (전역)
surface = None

def camera_callback(image, frame_queue, inference_enabled):
    """
    CARLA 카메라 센서가 새로운 프레임을 보낼 때 실행되는 콜백 함수
    - image: carla.Image 객체
    - frame_queue: base64 변환된 프레임을 넣을 큐
    - inference_enabled: 추론 활성화 여부
    """
    global surface
    # CARLA 이미지(raw_data)는 BGRA이므로 RGB로 변환
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    frame_rgb = arr[:, :, :3][:, :, ::-1]
    
    # Pygame Surface로 변환해 display에 출력 가능하게 함
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    surface = frame_surface

    # 추론 ON일 경우 frame_queue에 base64 인코딩된 이미지 넣기
    if inference_enabled:
        resized = np.array(Image.fromarray(frame_rgb).resize((224, 224)))
        b64_str = encode_frame_to_base64(resized)
        try:
            frame_queue.put_nowait(b64_str)
        except Full:
            pass
