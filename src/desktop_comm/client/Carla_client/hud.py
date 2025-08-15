# hud.py
import pygame
from config import WIDTH, HEIGHT

class HUD:
    def __init__(self, width, height, vehicle=None):
        self.width = width
        self.height = height
        self.base_width = WIDTH
        self.base_height = HEIGHT
        pygame.font.init()

        base_font_size = 16
        self.scale_factor = min(width / self.base_width, height / self.base_height)
        self.font_size = max(int(base_font_size * self.scale_factor), 8)

        self._font_mono = pygame.font.SysFont('consolas', self.font_size)
        self.info_font_size = max(int(18 * self.scale_factor), 12)
        self._info_font = pygame.font.SysFont('arial', self.info_font_size, bold=True)

        self._clock = pygame.time.Clock()
        self._frame = 0
        self._info_text = []
        self.vehicle = vehicle
        self.current_weather_name = "ClearNoon"

    def update_scale(self, width, height):
        self.width = width
        self.height = height
        self.scale_factor = min(width / self.base_width, height / self.base_height)

        base_font_size = 14
        new_font_size = max(int(base_font_size * self.scale_factor), 8)
        if new_font_size != self.font_size:
            self.font_size = new_font_size
            self._font_mono = pygame.font.SysFont('consolas', self.font_size)

        new_info_font_size = max(int(18 * self.scale_factor), 12)
        if new_info_font_size != self.info_font_size:
            self.info_font_size = new_info_font_size
            self._info_font = pygame.font.SysFont('arial', self.info_font_size, bold=True)

    def tick(self, world, latest_label=None, latest_conf=0.0,
             autopilot_enabled=False, reverse_mode=False, inference_enabled=False):
        """HUD 내부 상태를 업데이트"""
        self._frame = (self._frame + 1) % 30
        self._clock.tick()

        try:
            velocity = self.vehicle.get_velocity() if self.vehicle else None
            if velocity:
                speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                speed_str = f"Speed: {speed:.1f} km/h"
            else:
                speed_str = "Speed: N/A"
        except:
            speed_str = "Speed: N/A"

        self._info_text = [
            f"Client FPS: {self._clock.get_fps():.0f}",
            "-" * 19,
            "   Controls",
            "-" * 19,
            "[W] Forward",
            "[S] Brake",
            "[A] Left",
            "[D] Right",
            "[R] Reverse",
            "[P] Autopilot",
            "[I] Inference",
            "[F] Fullscreen",
            "[1~8] Weather",
            "[N] Reset",
            "[ESC] Quit",
            "-" * 19,
            "   Status",
            "-" * 19,
            f"Gear: {'Reverse' if reverse_mode else 'Forward'}",
            f"Autopilot: {'ON' if autopilot_enabled else 'OFF'}",
            f"Inference: {'ON' if inference_enabled else 'OFF'}",
            speed_str,
            f"Weather: {self.current_weather_name}"
        ]

        if latest_label and inference_enabled:
            self._info_text.append(f"Label: {latest_label}")
            self._info_text.append(f"Confidence: {latest_conf*100:.1f}%")

    def wrap_text(self, text, max_width):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test_line = current + (word + " ")
            if self._font_mono.size(test_line)[0] > max_width:
                lines.append(current.rstrip())
                current = word + " "
            else:
                current = test_line
        if current:
            lines.append(current.rstrip())
        return lines

    def render_top_right_info(self, display, latest_label, latest_conf, inference_enabled):
        fps_text = f"FPS: {self._clock.get_fps():.0f}"
        if latest_label and inference_enabled:
            prediction_text = f"Prediction: {latest_label}"
            confidence_text = f"Confidence: {latest_conf*100:.1f}%"
        else:
            prediction_text = "Prediction: --"
            confidence_text = "Confidence: --"

        text_color = (255, 255, 0)
        margin = int(8 * self.scale_factor)
        line_height = int(22 * self.scale_factor)
        panel_x = self.width - margin
        panel_y = margin

        fps_surf = self._info_font.render(fps_text, True, text_color)
        display.blit(fps_surf, fps_surf.get_rect(topright=(panel_x, panel_y)))

        pred_surf = self._info_font.render(prediction_text, True, text_color)
        display.blit(pred_surf, pred_surf.get_rect(topright=(panel_x, panel_y + line_height)))

        conf_surf = self._info_font.render(confidence_text, True, text_color)
        display.blit(conf_surf, conf_surf.get_rect(topright=(panel_x, panel_y + line_height * 2)))

    def render(self, display, latest_label, latest_conf, inference_enabled):
        current_width, current_height = display.get_size()
        self.update_scale(current_width, current_height)

        base_hud_width = 170
        hud_width = int(base_hud_width * self.scale_factor)
        hud_width = max(hud_width, 150)

        info_surface = pygame.Surface((hud_width, current_height))
        info_surface.set_alpha(120)
        info_surface.fill((0, 0, 0))
        display.blit(info_surface, (0, 0))

        base_line_height = 16
        line_height = max(int(base_line_height * self.scale_factor), 12)
        margin = int(8 * self.scale_factor)
        v_offset = margin

        for item in self._info_text:
            if item.startswith("Confidence") and latest_label is None:
                continue
            wrapped = self.wrap_text(item, hud_width - (margin * 2))
            for part in wrapped:
                color = (255, 255, 255)
                if part.startswith("Autopilot:"):
                    color = (0, 200, 0) if "ON" in part else (200, 0, 0)
                elif part.startswith("Inference:"):
                    color = (0, 200, 0) if "ON" in part else (200, 0, 0)
                elif part.startswith("Label:"):
                    color = (255, 255, 0)
                elif part.startswith("Confidence:"):
                    color = (255, 255, 0)
                display.blit(self._font_mono.render(part, True, color), (margin, v_offset))
                v_offset += line_height

        self.render_top_right_info(display, latest_label, latest_conf, inference_enabled)
