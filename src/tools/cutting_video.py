import cv2
import os
import glob

# 입력 폴더(여러 영상이 있음)
input_dir = '/home/kyj28/workspace/Road_Vision/Road_Vision_Data/downloads/normal_road_night'
# 결과 저장 폴더(상위 폴더)
output_dir = '/home/kyj28/workspace/Road_Vision/Road_Vision_Data/downloads/normal_road_night_cut'

# 결과 폴더 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 20분 = 1200초
segment_duration_sec = 20 * 60

# 입력폴더 아래의 모든 mp4 파일 구하기
video_files = sorted(glob.glob(os.path.join(input_dir, '*.mp4')))

for video_path in video_files:
    basename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    seg_frames = int(fps * segment_duration_sec)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    frame_idx = 0
    seg_idx = 1
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 분할 시작
        if frame_idx % seg_frames == 0:
            if out is not None:
                out.release()
            part_name = f"{basename}_part{seg_idx:02d}.mp4"
            out_path = os.path.join(output_dir, part_name)
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            print(f"Saving: {out_path}")
            seg_idx += 1

        out.write(frame)
        frame_idx += 1

    if out is not None:
        out.release()
    cap.release()

print("모든 영상의 분할/저장이 완료됐습니다!")
