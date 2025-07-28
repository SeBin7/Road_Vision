# video_timeline_infer.py
# ───────────────────────────────────────────────────────────────────────────────
# 동영상 특정 구간(타임라인)에서 도로 상태를 추론하기 전체 코드
#  • 트랙바 3개(타임라인, 시작, 끝)로 관심 구간 선택
#  • 실시간/일시정지/단프레임 이동·구간 이동 단축키
#  • 선택 구간 안에서만 슬라이딩 윈도우(10프레임) 추론 수행
# ───────────────────────────────────────────────────────────────────────────────
import os
import cv2
import glob
import time
import torch
import torch.nn.functional as F
from PIL import Image
from collections import deque
from torchvision import transforms

from CNN import CNNFeatureExtractor      # CNN backbone
from GRU_MLP import GRU_MLP_Classifier        # GRU-MLP classifier

# ──────────────────────────── 하이퍼파라미터 ────────────────────────────
SEQ_LEN = 10                 # 슬라이딩 윈도우 길이
WINDOW_NAME = 'Road-Vision'  # OpenCV 창 이름
BAR_COLOR  = (  0,255,  0)   # (B,G,R) – 예측 텍스트 색
FONT       = cv2.FONT_HERSHEY_SIMPLEX

# 클래스 인덱스 → 이름
label_map = {0: 'ice_road',
             1: 'normal_road',
             2: 'wet_road'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────── 추론 클래스 ──────────────────────────────
class VideoInference:
    def __init__(self, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.buffer  = deque(maxlen=seq_len)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
        ])

        # 모델 로드
        self.cnn = CNNFeatureExtractor().to(device)
        self.cls = GRU_MLP_Classifier(feature_dim=128).to(device)
        self.cnn.load_state_dict(torch.load('./best_cnn_feature_extractor.pth',
                                            map_location=device))
        self.cls.load_state_dict(torch.load('./best_gru_mlp_classifier.pth',
                                            map_location=device))
        self.cnn.eval(); self.cls.eval()

    # 단일 프레임 버퍼링
    def _push_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.buffer.append(self.transform(pil))

    # 버퍼가 가득 찼을 때 예측 반환
    def predict(self):
        if len(self.buffer) < self.seq_len:
            return None, None

        x = torch.stack(list(self.buffer)).unsqueeze(0)   # (1,T,C,H,W)
        b,t,c,h,w = x.size()
        x = x.view(b*t, c, h, w).to(device)

        with torch.no_grad():
            feat = self.cnn(x, flatten=True).view(b, t, -1)
            logit = self.cls(feat)
            
            prob  = F.softmax(logit, dim=1)[0]
            idx   = torch.argmax(prob).item()
            print(f"Predicted: {idx} → {label_map[idx]}")
            return label_map[idx], prob[idx].item()*100


# ──────────────────────────── 유틸리티 ────────────────────────────────
def frames_to_time(frames, fps):
    """정수 프레임 → 'HH:MM:SS' 문자열"""
    secs = int(frames / fps)
    return time.strftime('%H:%M:%S', time.gmtime(secs))


# ──────────────────────────── 메인 로직 ────────────────────────────────
def main(video_path: str):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f'동영상 파일을 찾을 수 없습니다: {video_path}')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'동영상을 열 수 없습니다: {video_path}')

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    # 초기 구간 (전체)
    start_frame = 0
    end_frame   = total - 1
    current     = 0
    paused      = False

    # 추론 모듈
    infer = VideoInference()

    # ── OpenCV 윈도우 & 트랙바 ─────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME)

    def on_timeline(pos):
        nonlocal current
        current = pos
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def on_start(pos):
        nonlocal start_frame
        start_frame = min(pos, end_frame-1)
        cv2.setTrackbarPos('Start', WINDOW_NAME, start_frame)

    def on_end(pos):
        nonlocal end_frame
        end_frame = max(pos, start_frame+1)
        cv2.setTrackbarPos('End', WINDOW_NAME, end_frame)

    cv2.createTrackbar('Timeline', WINDOW_NAME, 0, total-1, on_timeline)
    cv2.createTrackbar('Start',    WINDOW_NAME, 0, total-1, on_start)
    cv2.createTrackbar('End',      WINDOW_NAME, total-1, total-1, on_end)

    # ── 재생 루프 ─────────────────────────────────────────────────────
    print('❚❚ 스페이스: 재생/일시정지 | A/D: ±1프레임 | S/E: 구간점 이동')
    print('R: 구간 추론 활성 | Q: 종료\n')

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or current > end_frame:
                break
            current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            cv2.setTrackbarPos('Timeline', WINDOW_NAME, current)

            # 선택 구간이면 추론
            if start_frame <= current <= end_frame:
                infer._push_frame(frame)
                label, conf = infer.predict()
                if label:
                    txt = f'{label}: {conf:.1f}%'
                    cv2.putText(frame, txt, (10,35), FONT, 1, BAR_COLOR, 2,
                                cv2.LINE_AA)

            # 하단에 시간 정보 표시
            t_now  = frames_to_time(current, fps)
            t_tot  = frames_to_time(total,   fps)
            t_win  = f'[{frames_to_time(start_frame,fps)} - ' \
                     f'{frames_to_time(end_frame,fps)}]'
            info   = f'{t_now}/{t_tot}  {t_win}'
            cv2.putText(frame, info, (10,65), FONT, 0.6, (255,255,255), 1,
                        cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)

        # ── 키보드 이벤트 ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):                      # 재생/정지
            paused = not paused
        elif key in (ord('a'), 81):              # ← 1프레임 뒤로
            pos = max(current-1, 0); on_timeline(pos)
        elif key in (ord('d'), 83):              # → 1프레임 앞으로
            pos = min(current+1, total-1); on_timeline(pos)
        elif key == ord('s'):                    # 구간 시작점
            on_timeline(start_frame)
        elif key == ord('e'):                    # 구간 끝점
            on_timeline(end_frame)
        elif key == ord('r'):                    # 구간 추론 재시작
            infer.buffer.clear(); on_timeline(start_frame)
            paused = False
            print(f'▶ 구간 {frames_to_time(start_frame,fps)}'
                  f' ~ {frames_to_time(end_frame,fps)} 추론 시작')
        elif key == ord('q'):                    # 종료
            break

    cap.release()
    cv2.destroyAllWindows()


# ──────── CLI ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Road-Vision timeline inference')
    parser.add_argument('video', help='입력 동영상 파일(.mp4, .avi …)')
    args = parser.parse_args()

    main(args.video)
