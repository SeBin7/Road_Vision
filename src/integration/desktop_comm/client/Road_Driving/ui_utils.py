# ui_utils.py
import cv2
import numpy as np

def _pt(img, text, org, scale, color, thick=2):
    """OpenCV putText 래퍼 함수"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_prob_bars(img, probs, labels, topright=(0.72, 0.15), size=(0.25, 0.7)):
    """분류 확률을 막대 그래프로 시각화하는 함수"""
    h, w = img.shape[:2]
    x0, y0 = int(w * topright[0]), int(h * topright[1])
    bw, bh = int(w * size[0]), int(h * size[1])
    n = len(probs)
    pad = int(bh * 0.02)
    row_h = max(1, int((bh - pad * (n + 1)) // max(n, 1)))
    x1 = x0 + bw
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y0 + bh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    for i, (p, name) in enumerate(zip(probs, labels)):
        y = y0 + pad + i * (row_h + pad)
        bar_w = int((bw - 20) * float(p))
        cv2.rectangle(img, (x0 + 10, y), (x0 + 10 + bar_w, y + row_h), (60, 220, 255), -1)
        _pt(img, f"{name:<12} {p*100:5.1f}%", (x0 + 12, y + int(row_h * 0.8)),
            max(h / 1400, 0.6), (255, 255, 255), max(1, int(h / 900)))

def draw_info_panel(img, **kwargs):
    """다양한 실행 정보를 좌측 상단에 패널 형태로 표시하는 함수"""
    h, w = img.shape[:2]
    px, py = int(w * 0.03), int(h * 0.05)
    lh1, lh2 = max(2, int(h / 540)), max(1, int(h / 900))
    s1, s2 = h / 900, h / 1400
    box_w, box_h = int(w * 0.50), int(h * 0.22)
    overlay = img.copy()
    cv2.rectangle(overlay, (px - 10, py - 30), (px - 10 + box_w, py - 30 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    label = kwargs.get('label')
    conf = kwargs.get('conf')
    
    # conf 값이 None이 아닐 때만 예측 텍스트를 포맷팅
    if label and conf is not None:
        pred_text = f"{label}: {conf:.1f}%"
    else:
        pred_text = "N/A"

    info_texts = {
        'disp_fps': f"disp FPS: {kwargs.get('fps', 0):.2f}",
        'src': f"Src: {kwargs.get('src', 'N/A')}",
        'settings': f"BATCH={kwargs.get('batch', 'N/A')}  JPEG_Q={kwargs.get('jpeg_q', 'N/A')}",
        'prediction': pred_text,
        'server': f"{kwargs.get('server_url', '')}",
        'queues': f"Q(cap/enc/feat): {kwargs.get('q_sizes', '0/0/0')}"
    }

    _pt(img, info_texts['disp_fps'], (px, py), s1, (0, 200, 0), lh1)
    _pt(img, info_texts['src'], (px, py + int(h * 0.05)), s2, (200, 255, 200), lh2)
    _pt(img, info_texts['settings'], (px, py + int(h * 0.09)), s2, (255, 255, 0), lh2)
    _pt(img, info_texts['prediction'], (px, py + int(h * 0.13)), s1, (50, 200, 255), lh1)
    _pt(img, info_texts['server'], (px, py + int(h * 0.17)), s2, (180, 180, 255), lh2)
    _pt(img, info_texts['queues'], (px, py + int(h * 0.21)), s2, (200, 200, 200), lh2)