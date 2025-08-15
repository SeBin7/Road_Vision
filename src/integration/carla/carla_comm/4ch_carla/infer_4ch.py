# ───────────────────────────────────────────────────────────────────────────────
# video_timeline_infer.py  ── 4K-friendly display 버전
# ───────────────────────────────────────────────────────────────────────────────
import os, cv2, glob, time, torch, torch.nn.functional as F
from PIL import Image
from collections import deque
from torchvision import transforms

from Mobilenet_hailo_4ch_carla import MobileNetFeatureExtractor
from GRU_MLP_xpu import GRU_MLP_Classifier_XPU as GRU

import torch, intel_extension_for_pytorch as ipex

# ─────────────── 하이퍼파라미터 ───────────────
SEQ_LEN      = 10
WINDOW_NAME  = 'Road-Vision'
BAR_COLOR    = (  0,255,  0)  # BGR
FONT         = cv2.FONT_HERSHEY_SIMPLEX
MAX_SCREEN_H = 1080           # 모니터 세로 해상도(필요시 수정)

label_map = {0: 'normal_road', 1: 'wet_road'}
#label_map = {0: 'normal_road', 1: 'snow_road', 2: 'wet_road'}  # 예시로 수정
device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

# ─────────────── 동적 텍스트 파라미터 ───────────────
def get_text_params(h):
    """프레임 높이에 비례한 (폰트스케일, 두께, 위치 y오프셋) 반환"""
    font_scale = h / 1080 * 1.0          # 1080p→1.0, 4K(2160p)→2.0
    thickness  = max(1, int(h / 1080 * 2))
    y_pred     = int(h * 0.04)           # 상태바 y위치
    y_time     = int(h * 0.08)           # 시간바 y위치
    return font_scale, thickness, y_pred, y_time

# ─────────────── 추론 클래스 ───────────────
class VideoInference:
    def __init__(self, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.buffer  = deque(maxlen=seq_len)
       
         # RGB만 ToTensor, 0-1 스케일. Normalize는 4채널 합친 뒤 처리
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

        self.cnn = MobileNetFeatureExtractor().to(device)
        self.cls = GRU(feature_dim=128).to(device)
        
        self.cnn.load_state_dict(torch.load('./cnn_feature_extractor_4ch_carla.pth',
                                            map_location=device))
        self.cls.load_state_dict(torch.load('./gru_mlp_classifier_4ch_carla.pth',
                                            map_location=device))
        self.cnn.eval()
        self.cls.eval()

    def _push_frame(self, frame_bgr):
        # RGB 변환 및 Tensor
         rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
         pil = Image.fromarray(rgb)
         t_rgb = self.transform(pil)  # (3,224,224)
         # Canny 채널 생성
         gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
         edges = cv2.Canny(gray, 50, 150)
         edges = cv2.resize(edges, (224,224))
         t_edge = torch.from_numpy(edges).unsqueeze(0).float().div(255.0)  # (1,224,224)
         # 4채널 결합 및 Normalize
         img4 = torch.cat([t_rgb, t_edge], dim=0)  # (4,224,224)
         mean = torch.tensor([0.485,0.456,0.406,0.5], device=img4.device)
         std  = torch.tensor([0.229,0.224,0.225,0.5], device=img4.device)
         img4 = (img4 - mean[:,None,None]) / std[:,None,None]
         self.buffer.append(img4)

    def predict(self):
        if len(self.buffer) < self.seq_len: return None, None
        x = torch.stack(list(self.buffer)).unsqueeze(0)   # (1,T,4,H,W)
        b,t,c,h,w = x.size()
        with torch.no_grad():
            feat  = self.cnn(x.view(b*t,c,h,w).to(device)).view(b,t,-1)
            logit = self.cls(feat)
            prob  = F.softmax(logit, dim=1)[0]
            idx   = torch.argmax(prob).item()
            return label_map[idx], prob[idx].item()*100

# ─────────────── 유틸리티 ───────────────
def frames_to_time(frames, fps):
    secs = int(frames / fps)
    return time.strftime('%H:%M:%S', time.gmtime(secs))

# ─────────────── 메인 루틴 ───────────────
def main(video_path: str):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame, end_frame, current, paused = 0, total-1, 0, False
    infer = VideoInference()

    # ❶ 창 크기 자유 + 비율 유지
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    cv2.resizeWindow(WINDOW_NAME, 1920, 1080) # 모니터 화면 크기 지정 1080p

    # cv2.setWindowProperty(WINDOW_NAME,
    #                   cv2.WND_PROP_FULLSCREEN,
    #                   cv2.WINDOW_FULLSCREEN) # 전체 화면 모드

    # ❷ 트랙바 콜백 정의
    def on_timeline(pos): cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    def on_start(pos):
        nonlocal start_frame; start_frame = min(pos, end_frame-1)
        cv2.setTrackbarPos('Start', WINDOW_NAME, start_frame)
    def on_end(pos):
        nonlocal end_frame; end_frame = max(pos, start_frame+1)
        cv2.setTrackbarPos('End', WINDOW_NAME, end_frame)

    cv2.createTrackbar('Timeline', WINDOW_NAME, 0, total-1, on_timeline)
    cv2.createTrackbar('Start',    WINDOW_NAME, 0, total-1, on_start)
    cv2.createTrackbar('End',      WINDOW_NAME, total-1, total-1, on_end)

    print('❚❚ Space:재생/일시정지 | A/D:±1프레임 | S/E:구간점 이동 | R:재추론 | Q:종료')

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or current > end_frame: break
            current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            cv2.setTrackbarPos('Timeline', WINDOW_NAME, current)

            # ❸ 추론 버퍼 갱신
            if start_frame <= current <= end_frame:
                infer._push_frame(frame)
                label, conf = infer.predict()

            # ❹ ------- DISPLAY 영역 (해상도 대응) -----------------
            h, w = frame.shape[:2]
            # 최대 화면 세로 `MAX_SCREEN_H`에 맞춰 축소 (예: 모니터 1080p → 0.5 스케일)
            disp_scale = min(1.0, MAX_SCREEN_H / h)
            display_frame = cv2.resize(frame, (int(w*disp_scale), int(h*disp_scale)))

            dh = display_frame.shape[0]  # 축소 후 높이
            font_scale, thickness, y_pred, y_time = get_text_params(dh)

            if label:
                txt = f'{label}: {conf:.1f}%'
                cv2.putText(display_frame, txt, (int(dh*0.03), y_pred),
                            FONT, font_scale, BAR_COLOR, thickness, cv2.LINE_AA)

            info = f'{frames_to_time(current,fps)}/{frames_to_time(total,fps)}  ' \
                   f'[{frames_to_time(start_frame,fps)} - {frames_to_time(end_frame,fps)}]'
            cv2.putText(display_frame, info, (int(dh*0.03), y_time),
                        FONT, font_scale*0.7, (255,255,255), thickness, cv2.LINE_AA)
            # -------------------------------------------------------

            cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '): paused = not paused
        elif key in (ord('a'), 81):  on_timeline(max(current-1, 0))
        elif key in (ord('d'), 83):  on_timeline(min(current+1, total-1))
        elif key == ord('s'):        on_timeline(start_frame)
        elif key == ord('e'):        on_timeline(end_frame)
        elif key == ord('r'):
            infer.buffer.clear(); on_timeline(start_frame); paused=False
            print(f'▶ 구간 재추론: {frames_to_time(start_frame,fps)} ~ {frames_to_time(end_frame,fps)}')
        elif key == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

# ─────────────── CLI ───────────────
if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser(description='Road-Vision timeline inference')
    parser.add_argument('video', help='입력 동영상 파일(.mp4, .avi …)')
    args = parser.parse_args()
    main(args.video)
