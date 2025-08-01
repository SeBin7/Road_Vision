import cv2
import numpy as np
import base64
import requests
import torch
import torch.nn.functional as F
from collections import deque
from GRU_MLP_xpu import GRU_MLP_Classifier_XPU as GRU

# ==== 설정 값 ====
VIDEO_PATH = './normal_road.mp4'  # 추론 대상 동영상 경로
SERVER_URL = 'http://192.168.100.147:8001/infer'  # 라즈베리파이 추론 서버 주소
SEQ_LEN = 10  # GRU 시퀀스 길이
FEATURE_DIM = 128  # feature vector 차원
NUM_CLASSES = 4  # 분류할 도로 상태 클래스 수
MODEL_PATH = './best_gru_mlp_classifier.pth'  # GRU+MLP 모델 경로

# ==== 디바이스 설정 ====
DEVICE = torch.device('xpu' if torch.xpu.is_available() else 'cpu')  # Intel XPU 사용

# ==== 클래스 라벨 맵 ====
label_map = {0: 'broken', 1: 'normal_road', 2: 'snow_road', 3: 'wet_road'}

# ==== GRU + MLP 분류기 로드 ====
model = GRU(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==== 시퀀스 버퍼 및 동영상 설정 ====
buffer = deque(maxlen=SEQ_LEN)  # 시퀀스 저장 버퍼
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_idx = 0

# ==== 메인 루프: 프레임 단위 처리 ====
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 영상 종료 시 루프 탈출

    # 프레임 → JPEG 인코딩 → base64 문자열 변환
    _, img_encoded = cv2.imencode('.jpg', frame)
    b64_frame = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

    # 서버에 보낼 payload
    payload = {
        'frame_base64': b64_frame,
        'frame_idx': frame_idx
    }
    headers = {'Content-Type': 'application/json'}

    try:
        # POST 요청으로 추론 서버에 전송
        resp = requests.post(SERVER_URL, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # 응답 체크
        if 'feature_vector_base64' not in data or 'frame_idx' not in data:
            print(f"[WARN] 잘못된 응답 데이터: {data}")
            frame_idx += 1
            continue

        # 프레임 인덱스 불일치 경고
        if data['frame_idx'] != frame_idx:
            print(f"[WARN] frame_idx 불일치 (요청 {frame_idx} / 응답 {data['frame_idx']})")

        # feature vector 복원 및 정규화 (uint8 → float32)
        feature_bytes = base64.b64decode(data['feature_vector_base64'])
        feature_vector_uint8 = np.frombuffer(feature_bytes, dtype=np.uint8)
        feature_vector = feature_vector_uint8.astype(np.float32) / 255.0
        buffer.append(torch.tensor(feature_vector, dtype=torch.float32).to(DEVICE))

        label, conf = None, None
        # 시퀀스가 꽉 찼을 때 추론 수행
        if len(buffer) == SEQ_LEN:
            seq_x = torch.stack(list(buffer)).unsqueeze(0)  # (1, 10, 128)
            with torch.no_grad():
                logit = model(seq_x)
                prob = F.softmax(logit, dim=1)[0]  # 확률 분포
                idx = torch.argmax(prob).item()
                label = label_map[idx]
                conf = prob[idx].item() * 100

        # ==== 시각화 ====
        disp_frame = frame.copy()
        h, w = disp_frame.shape[:2]

        # 도로 상태 분류 결과 출력
        if label:
            text = f"{label}: {conf:.1f}%"
            cv2.putText(disp_frame, text, (int(w * 0.03), int(h * 0.08)),
                        cv2.FONT_HERSHEY_SIMPLEX, h / 1080, (0, 255, 0), max(2, int(h / 540)),
                        cv2.LINE_AA)

        # 영상 시간 정보 출력 (현재/전체 시간)
        time_info = f'{int(frame_idx/fps)//60}:{int(frame_idx/fps)%60:02}/{int(total_frames/fps)//60}:{int(total_frames/fps)%60:02}'
        cv2.putText(disp_frame, time_info, (int(w * 0.7), int(h * 0.08)), cv2.FONT_HERSHEY_SIMPLEX,
                    h / 1400, (255, 255, 255), max(1, int(h / 1080)), cv2.LINE_AA)

        # 영상 출력
        cv2.imshow('Distributed Road-Vision', disp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Q 키로 중단

    # ==== 예외 처리 ====
    except requests.exceptions.ConnectionError as ce:
        print(f"[ERROR] 통신 오류 (서버 연결 실패): {ce}")
        break
    except requests.exceptions.Timeout as te:
        print(f"[ERROR] 요청 타임아웃: {te}")
    except Exception as e:
        print(f"[ERROR] 처리 오류: {e}")

    frame_idx += 1

# 자원 정리
cap.release()
cv2.destroyAllWindows()
