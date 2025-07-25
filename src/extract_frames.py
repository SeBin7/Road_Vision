import cv2
import os
import argparse
from collections import deque
import numpy as np
from pathlib import Path

def extract_sequences_from_video(video_path, output_dir, seq_len=5, stride=1):
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
            # 저장 디렉토리 생성
            seq_path = os.path.join(output_dir, f"seq_{seq_count:05d}")
            os.makedirs(seq_path, exist_ok=True)

            for i, f in enumerate(frame_queue):
                frame_file = os.path.join(seq_path, f"frame_{i}.jpg")
                cv2.imwrite(frame_file, f)

            print(f"Saved sequence {seq_count} → {seq_path}")
            seq_count += 1

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="input video path")
    parser.add_argument("--out", type=str, required=True, help="output dir to save frame sequences")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()

    extract_sequences_from_video(args.video, args.out, args.seq_len, args.stride)