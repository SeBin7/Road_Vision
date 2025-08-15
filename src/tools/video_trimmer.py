import cv2

cap = cv2.VideoCapture("data/videos/sample.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_limit = int(fps * 20)  # 20초

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("data/videos/sample_short.mp4", fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or count >= frame_limit:
        break
    out.write(frame)
    count += 1

cap.release()
out.release()