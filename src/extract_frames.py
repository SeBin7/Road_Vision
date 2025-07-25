import os
import cv2
import glob
from collections import deque

def extract_sequences_from_video(video_path, output_root, seq_len=5, stride=1):
    """
    단일 영상에서 시퀀스 단위로 프레임 저장
    각 시퀀스는 하나의 폴더(예: seq_00001/)로 저장됨
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    frame_queue = deque(maxlen=seq_len)
    frame_count = 0
    seq_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_queue.append(frame)
        frame_count += 1

        if len(frame_queue) == seq_len and (frame_count - seq_len) % stride == 0:
            seq_path = os.path.join(output_dir, f"seq_{seq_count:05d}")
            os.makedirs(seq_path, exist_ok=True)

            for i, f in enumerate(frame_queue):
                frame_file = os.path.join(seq_path, f"frame_{i}.jpg")
                cv2.imwrite(frame_file, f)

            print(f"[{video_name}] Saved sequence {seq_count} → {seq_path}")
            seq_count += 1

    cap.release()
    print(f"[{video_name}] ▶ 총 {seq_count}개의 시퀀스 저장 완료")


def extract_sequences_batch(input_dir, output_root, seq_len=5, stride=1, exts=[".mp4", ".avi", ".mov"]):
    """
    input_dir의 모든 영상에 대해 시퀀스 추출 수행
    """
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))

    print(f"총 {len(video_files)}개의 영상 처리 시작...")
    for video_path in video_files:
        extract_sequences_from_video(video_path, output_root, seq_len, stride)
    print("🎉 모든 영상 처리 완료.")

# 예: 직접 실행 시
if __name__ == "__main__":
    extract_sequences_batch(
        input_dir="../data/videos",        # 입력: 여러 영상
        output_root="../data/frames",       # 출력: 시퀀스 저장 경로
        seq_len=5,
        stride=1
    )

#update