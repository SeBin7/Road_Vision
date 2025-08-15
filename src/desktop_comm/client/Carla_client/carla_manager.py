# carla_manager.py
import carla
import random
from config import CARLA_HOST, CARLA_PORT, WORLD_NAME, weather_presets

def connect_carla():
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.load_world(WORLD_NAME)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.03
    world.apply_settings(settings)

    return client, world

def spawn_vehicle_and_camera(world, bp_lib, camera_callback, frame_queue, inference_enabled, width, height):
    spawn_points = world.get_map().get_spawn_points()
    vehicle = None
    for sp in spawn_points:
        try:
            vehicle = world.spawn_actor(bp_lib.filter('vehicle.*')[0], sp)
            break
        except RuntimeError:
            continue
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle")

    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(width))
    cam_bp.set_attribute('image_size_y', str(height))
    cam_bp.set_attribute('fov', '90')
    
    camera = world.spawn_actor(cam_bp,
                               carla.Transform(carla.Location(x=1.5, z=1.7)),
                               attach_to=vehicle)
    camera.listen(lambda image: camera_callback(image, frame_queue, inference_enabled))

    return vehicle, camera

def set_weather(world, key):
    if key in weather_presets:
        world.set_weather(weather_presets[key])

def toggle_autopilot(vehicle, enabled):
    TM_PORT = 8000
    vehicle.set_autopilot(enabled, TM_PORT)

def destroy_actor(actor):
    if actor:
        if hasattr(actor, 'stop'):
            try:
                actor.stop()
            except Exception as e:
                print(f"Warning stopping actor: {e}")
        actor.destroy()
