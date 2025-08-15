# config.py
import os

def get_config():
    """프로젝트의 모든 설정을 담은 딕셔너리를 반환합니다."""
    return {
        # ── 서버 및 모델 설정 ──────────────────────────────────
        "SERVER_URL": "http://0.0.0.0:5000/batch_extract_features_real_4ch",
        "CLS_WEIGHT": "./gru_mlp_classifier_4ch_val.pth",
        "LABEL_MAP": {0: 'broken', 1: 'normal_road', 2: 'snow_road', 3: 'wet_road'},

        # ── 모델 파라미터 ───────────────────────────────────────
        "SEQ_LEN": 10,
        "FEATURE_DIM": 128,
        "NUM_CLASSES": 4,
        "DEQUANT_MODE": os.getenv("DEQUANT_MODE", "sym"),  # "sym" 또는 "unit"

        # ── 파이프라인 및 성능 설정 ──────────────────────────────
        "BATCH": 32,
        "TARGET_HW": (224, 224),
        "JPEG_QUALITY": 100,
        "CAP_QUEUE": 60,
        "ENC_QUEUE": 60,
        
        # ── 재생 및 화면 설정 ───────────────────────────────────
        "SYNC_TO_SRC_FPS": True,
        "LOOP_PLAYBACK": True,
        "FULLSCREEN": True,
        "DISP_SIZE": (1280, 720),
        
        # ── 멀티 모니터 설정 ────────────────────────────────────
        "DISPLAY_GEOMS": [
            (0, 0, 1920, 1080),       # 모니터 1 (메인)
            (1920, 0, 1920, 1080),    # 모니터 2 (서브)
        ],
        "TARGET_DISPLAY": 1,  # 0: 메인, 1: 서브
    }