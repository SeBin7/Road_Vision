#grad_cam.py -> show the heat map
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from Mobilenet import MobileNetFeatureExtractor

# 모델 설정
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"사용 디바이스: {device}")
model = MobileNetFeatureExtractor().to(device)
model.load_state_dict(torch.load("./best_cnn_feature_extractor.pth", map_location=device))
model.eval()

target_layer = model.backbone[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 영상 설정
video_path = "../test_videos/snow_road/snow_road02.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
display_width, display_height = 960, 540
bar_height = 20

# 슬라이더 상태 변수
current_frame = 0
seek_request = False  # 트랙바 조작 시 True

def on_trackbar(val):
    global current_frame, seek_request
    current_frame = val
    seek_request = True

cv2.namedWindow("Grad-CAM Seekable")
cv2.createTrackbar("Seek", "Grad-CAM Seekable", 0, total_frames - 1, on_trackbar)

while cap.isOpened():
    if seek_request:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        seek_request = False
    else:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    grayscale_cam = cam(input_tensor=input_tensor)[0]
    cam_image = show_cam_on_image(rgb / 255.0, grayscale_cam, use_rgb=True)
    cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cam_display = cv2.resize(cam_bgr, (display_width, display_height))

    # 재생 바 추가
    canvas = np.zeros((display_height + bar_height, display_width, 3), dtype=np.uint8)
    canvas[:display_height] = cam_display

    progress_ratio = current_frame / total_frames
    progress_width = int(display_width * progress_ratio)
    cv2.rectangle(canvas, (0, display_height), (display_width, display_height + bar_height), (50, 50, 50), -1)
    cv2.rectangle(canvas, (0, display_height), (progress_width, display_height + bar_height), (0, 200, 0), -1)

    # 현재 위치 슬라이더 동기화
    cv2.setTrackbarPos("Seek", "Grad-CAM Seekable", current_frame)

    cv2.imshow("Grad-CAM Seekable", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()