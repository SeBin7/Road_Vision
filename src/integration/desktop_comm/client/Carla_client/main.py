import sys
import ctypes
import random
import threading
from collections import deque
from queue import Queue, Empty, Full

import pygame
import torch
import carla
import GRU_MLP_xpu

from config import (
    WIDTH, HEIGHT, FULLSCREEN_WIDTH, FULLSCREEN_HEIGHT, FPS, SEQ_LEN,
    FEATURE_DIM_MODEL, MODEL_PATH, LABEL_MAP, weather_presets,
    CARLA_HOST, CARLA_PORT, WORLD_NAME
)
from utils import gru_infer, post_batch_images_to_edge
from hud import HUD
import camera_manager  # camera_callback, surface 포함
from inference_workers import server_worker, infer_worker
from carla_manager import connect_carla, spawn_vehicle_and_camera, set_weather, toggle_autopilot, destroy_actor

if sys.platform == 'win32':
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass

# 전역 상태 변수 선언
autopilot_enabled = False
reverse_mode = False
inference_enabled = True
latest_label_conf = {'label': None, 'conf': 0.0}
hud_lock = threading.Lock()

def main():
    global autopilot_enabled, reverse_mode, inference_enabled

    # CARLA 서버 연결 및 초기 환경 설정
    client, world = connect_carla()
    bp_lib = world.get_blueprint_library()
    TM_PORT = 8000

    # 처리용 큐 생성
    frame_queue = Queue(maxsize=30)
    feature_queue = Queue(maxsize=30)

    # 차량과 카메라 생성
    vehicle, camera = spawn_vehicle_and_camera(world, bp_lib, camera_manager.camera_callback, frame_queue, inference_enabled, WIDTH, HEIGHT)

    # GRU 모델 로드 (CPU 환경으로 적재)
    device = torch.device('cpu')
    GRU = GRU_MLP_xpu.GRU_MLP_Classifier_XPU
    gru_model = GRU(feature_dim=FEATURE_DIM_MODEL, hidden_dim=64, num_classes=2).to(device)
    gru_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    gru_model.eval()

    # Pygame 초기화 및 HUD 구성
    pygame.init()
    display = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CARLA with HUD - Async Inference")
    clock = pygame.time.Clock()

    hud = HUD(WIDTH, HEIGHT, vehicle)
    control = carla.VehicleControl()
    fullscreen_enabled = False

    # 서버 통신 및 추론 비동기 스레드 시작
    threading.Thread(target=server_worker, args=(frame_queue, feature_queue, SEQ_LEN), daemon=True).start()
    threading.Thread(target=infer_worker, args=(feature_queue, gru_model, device, SEQ_LEN, LABEL_MAP, hud_lock, latest_label_conf), daemon=True).start()

    print("▶ 조작 키 안내:")
    print("  WASD: 전진/후진/좌우 조향")
    print("  P: 자율주행 토글")
    print("  R: 후진 모드 토글")
    print("  I: 추론 기능 토글")
    print("  F: 전체화면 토글")
    print("  1~8: 날씨 변경")
    print("  N: 차량 재생성")
    print("  ESC: 종료")

    spawn_points = world.get_map().get_spawn_points()
    cam_bp = bp_lib.find('sensor.camera.rgb')

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    raise KeyboardInterrupt

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        autopilot_enabled = not autopilot_enabled
                        toggle_autopilot(vehicle, autopilot_enabled)
                        print(f"Autopilot {'ON' if autopilot_enabled else 'OFF'}")

                    elif event.key == pygame.K_r:
                        reverse_mode = not reverse_mode
                        print(f"Reverse mode {'ON' if reverse_mode else 'OFF'}")

                    elif event.key == pygame.K_i:
                        inference_enabled = not inference_enabled
                        print(f"Inference {'ON' if inference_enabled else 'OFF'}")

                    elif event.key == pygame.K_f:
                        fullscreen_enabled = not fullscreen_enabled
                        if fullscreen_enabled:
                            display = pygame.display.set_mode((FULLSCREEN_WIDTH, FULLSCREEN_HEIGHT), pygame.FULLSCREEN)
                        else:
                            display = pygame.display.set_mode((WIDTH, HEIGHT))
                        hud.update_scale(*display.get_size())
                        print(f"Fullscreen {'ON' if fullscreen_enabled else 'OFF'}")

                    elif event.key in weather_presets:
                        set_weather(world, event.key)
                        world.tick()  # 여기 추가!
                        print("Weather changed")

                    elif event.key == pygame.K_n:
                        print("[INFO] 차량 리스폰")
                        camera.stop()
                        destroy_actor(camera)
                        destroy_actor(vehicle)
                        with frame_queue.mutex:
                            frame_queue.queue.clear()
                        with feature_queue.mutex:
                            feature_queue.queue.clear()
                        with hud_lock:
                            latest_label_conf['label'], latest_label_conf['conf'] = None, 0.0

                        vehicle = world.try_spawn_actor(
                            bp_lib.filter('vehicle.*')[0],
                            random.choice(spawn_points)
                        )
                        if not vehicle:
                            raise RuntimeError("차량 스폰 실패")

                        camera = world.spawn_actor(cam_bp,
                                                  carla.Transform(carla.Location(x=1.5, z=2.4)),
                                                  attach_to=vehicle)
                        camera.listen(lambda image: camera_manager.camera_callback(image, frame_queue, inference_enabled))
                        hud.vehicle = vehicle

            # 수동 운전 처리
            if not autopilot_enabled:
                keys = pygame.key.get_pressed()
                control.throttle = 1.0 if keys[pygame.K_w] else 0.0
                control.brake = 1.0 if keys[pygame.K_s] else 0.0
                control.steer = -0.3 if keys[pygame.K_a] else (0.3 if keys[pygame.K_d] else 0.0)
                control.reverse = reverse_mode
                vehicle.apply_control(control)

            # 영상 출력부: 항상 camera_manager.surface 최신값 사용
            display.fill((0, 0, 0))
            if camera_manager.surface:
                current_w, current_h = display.get_size()
                surface_w, surface_h = camera_manager.surface.get_width(), camera_manager.surface.get_height()
                if (current_w, current_h) != (surface_w, surface_h):
                    scaled_surface = pygame.transform.scale(camera_manager.surface, (current_w, current_h))
                    display.blit(scaled_surface, (0, 0))
                else:
                    display.blit(camera_manager.surface, (0, 0))

            with hud_lock:
                hud.tick(world, latest_label_conf['label'], latest_label_conf['conf'], autopilot_enabled, reverse_mode, inference_enabled)
                hud.render(display, latest_label_conf['label'], latest_label_conf['conf'], inference_enabled)

            pygame.display.flip()
            clock.tick(FPS)
            world.tick()

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        camera.stop()
        destroy_actor(camera)
        destroy_actor(vehicle)
        pygame.quit()


if __name__ == "__main__":
    main()
