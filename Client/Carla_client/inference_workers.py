# inference_workers.py
from collections import deque
from queue import Empty, Full
from utils import post_batch_images_to_edge, gru_infer

def server_worker(frame_queue, feature_queue, seq_len):
    """
    frame_queue에서 base64 인코딩 이미지들을 받아 일정 수(SEQ_LEN) 쌓이면
    엣지 서버로 보내어 feature vector들을 받아 feature_queue에 저장한다.
    """
    buffer = []
    while True:
        try:
            b64_img = frame_queue.get(timeout=1)
            buffer.append(b64_img)
            if len(buffer) >= seq_len:
                features = post_batch_images_to_edge(buffer)
                buffer.clear()
                if features is not None:
                    for f in features:
                        try:
                            feature_queue.put_nowait(f)
                        except Full:
                            pass
        except Empty:
            continue


def infer_worker(feature_queue, gru_model, device, seq_len, LABEL_MAP, hud_lock, latest_label_conf):
    """
    feature_queue에서 feature vectors를 받아 시퀀스를 구성한 후
    GRU 모델로 추론을 수행해서 최신 레이블과 신뢰도 정보를 업데이트한다.
    """
    seq_buffer = deque(maxlen=seq_len)
    while True:
        try:
            feat = feature_queue.get(timeout=1)
            seq_buffer.append(feat)
            if len(seq_buffer) == seq_len:
                pred, conf = gru_infer(list(seq_buffer), gru_model, device)
                with hud_lock:
                    latest_label_conf['label'] = LABEL_MAP[pred]
                    latest_label_conf['conf'] = conf
                    print(f"Prediction: {LABEL_MAP[pred]}, Confidence: {conf*100:.2f}%")
        except Empty:
            continue
