# utils.py
import base64
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from config import EDGE_URL, FEATURE_DIM_SERVER, FEATURE_DIM_MODEL

def decode_feature_vector(feat_b64: str,
                          server_dim=FEATURE_DIM_SERVER,
                          model_dim=FEATURE_DIM_MODEL) -> np.ndarray:
    raw = base64.b64decode(feat_b64)
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size != server_dim:
        raise ValueError(f"Expected server feature dim {server_dim}, got {arr.size}")

    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    nan_count = np.sum(nan_mask)
    inf_count = np.sum(inf_mask)

    if nan_count > 0 or inf_count > 0:
        if (nan_count + inf_count) / arr.size > 0.1:
            print(f"[WARN] Feature vector contains NaN({nan_count}) / Inf({inf_count}) values")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    padded = np.zeros(model_dim, dtype=np.float32)
    padded[:server_dim] = arr
    return padded

def gru_infer(features_seq_b64: list, model: torch.nn.Module, device):
    decoded = [decode_feature_vector(f) for f in features_seq_b64]
    seq_arr = np.stack(decoded, axis=0)  # (seq_len, feature_dim)
    x = torch.from_numpy(seq_arr).unsqueeze(0).to(device)  # (1, seq_len, feature_dim)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1)[0]
        pred = torch.argmax(probs).item()
        conf = probs[pred].item()
    return pred, conf

def encode_frame_to_base64(frame_rgb):
    pil_img = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def post_batch_images_to_edge(b64_images):
    try:
        resp = requests.post(f"{EDGE_URL}/batch_extract_features",
                             json={"images": b64_images}, timeout=5)
        resp.raise_for_status()
        return resp.json().get("features", None)
    except Exception as e:
        print(f"POST error: {e}")
        return None
