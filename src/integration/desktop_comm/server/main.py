# test.py
import base64
import cv2
import numpy as np
import hailo_platform as hpf
import logging
from flask import Flask, request, jsonify

# ─────────────────────────────────────────────────────────────
# 기존 CarIa(3ch) — 그대로 유지
# ─────────────────────────────────────────────────────────────
HEF_PATH = './cnn_feature_extractor_3ch_carla.hef'
hef   = hpf.HEF(HEF_PATH)
device= hpf.VDevice()
cfg   = hpf.ConfigureParams.create_from_hef(hef, hpf.HailoStreamInterface.PCIe)
ng, = device.configure(hef, cfg)
ng_params = ng.create_params()
# ng.activate(ng_params)

in_info = hef.get_input_vstream_infos()[0]
out_info = hef.get_output_vstream_infos()[0]
in_name = in_info.name
out_name = out_info.name

# vstream params 1회 생성(재사용)
in_vs_carla  = hpf.InputVStreamParams.make_from_network_group(ng,  quantized=True, format_type=hpf.FormatType.UINT8)
out_vs_carla = hpf.OutputVStreamParams.make_from_network_group(ng, quantized=True, format_type=hpf.FormatType.UINT8)

# ─────────────────────────────────────────────────────────────
# 기존 Real(3ch) — 그대로 유지
# ─────────────────────────────────────────────────────────────
HEF_PATH_REAL = './cnn_feature_extractor_3ch_val_real.hef'
hef_real = hpf.HEF(HEF_PATH_REAL)
cfg_real = hpf.ConfigureParams.create_from_hef(hef_real, hpf.HailoStreamInterface.PCIe)
ng_real, = device.configure(hef_real, cfg_real)
ng_params_real = ng_real.create_params()
# ng_real.activate(ng_params_real)

in_info_real = hef_real.get_input_vstream_infos()[0]
out_info_real = hef_real.get_output_vstream_infos()[0]
in_name_real = in_info_real.name
out_name_real = out_info_real.name

in_vs_real  = hpf.InputVStreamParams.make_from_network_group(ng_real,  quantized=True, format_type=hpf.FormatType.UINT8)
out_vs_real = hpf.OutputVStreamParams.make_from_network_group(ng_real, quantized=True, format_type=hpf.FormatType.UINT8)

# ─────────────────────────────────────────────────────────────
# ★신규: Real(4ch) — 새 HEF 추가
# ──────────────────────────────────────────────────

HEF_PATH_REAL_4CH = './cnn_feature_extractor_4ch_val.hef'
hef_real4 = hpf.HEF(HEF_PATH_REAL_4CH)
cfg_real4 = hpf.ConfigureParams.create_from_hef(hef_real4, hpf.HailoStreamInterface.PCIe)
ng_real4, = device.configure(hef_real4, cfg_real4)
ng_params_real4 = ng_real4.create_params()
# ng_real4.activate(ng_params_real4)

in_info_real4 = hef_real4.get_input_vstream_infos()[0]
out_info_real4 = hef_real4.get_output_vstream_infos()[0]
in_name_real4 = in_info_real4.name
out_name_real4 = out_info_real4.name

in_vs_real4  = hpf.InputVStreamParams.make_from_network_group(ng_real4,  quantized=True, format_type=hpf.FormatType.UINT8)
out_vs_real4 = hpf.OutputVStreamParams.make_from_network_group(ng_real4, quantized=True, format_type=hpf.FormatType.UINT8)


def _get_format_type(vs):
    if isinstance(vs, dict):
        # dict: {stream_name: VStreamInfo}
        first_val = next(iter(vs.values()))
        return getattr(first_val, "format_type", "unknown")
    else:
        return getattr(vs, "format_type", "unknown")

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────────────────
# 공통 전처리 (기본: NCHW, UINT8) — HEF가 NHWC면 여기만 바꾸면 됨
# ─────────────────────────────────────────────────────────────
def _decode_to_bgr(jpeg_b64: str):
    data = base64.b64decode(jpeg_b64)
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('OpenCV decode failed')
    return img

def preprocess_3ch_bgr_uint8(jpeg_b64: str) -> np.ndarray:
    img = _decode_to_bgr(jpeg_b64)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.transpose(2,0,1)[None,...]  # (1,3,224,224) NCHW
    return np.ascontiguousarray(img, dtype=np.uint8)

def preprocess_4ch_bgr_uint8(jpeg_b64: str) -> np.ndarray:
    """RGB + Canny edge(1ch) → (1,4,224,224) UINT8 NCHW"""
    img_bgr = _decode_to_bgr(jpeg_b64)
    print(f"[LOG] BGR shape={img_bgr.shape}, dtype={img_bgr.dtype}, min={img_bgr.min()}, max={img_bgr.max()}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224,224))
    print(f"[LOG] RGB resized shape={img_rgb.shape}, dtype={img_rgb.dtype}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    print(f"[LOG] Canny edges shape={edges.shape}, dtype={edges.dtype}, edge_pixel_count={np.count_nonzero(edges)}")
    edges = cv2.resize(edges, (224,224))

    img4  = np.concatenate([img_rgb, edges[...,None]], axis=2)  # (224,224,4)
    print(f"[LOG] 4ch combined shape={img4.shape}, dtype={img4.dtype}, min={img4.min()}, max={img4.max()}")

    img4  = img4.transpose(2,0,1)[None,...]                     # (1,4,224,224)
    print(f"[LOG] Output shape={img4.shape}, dtype={img4.dtype}")

    return np.ascontiguousarray(img4, dtype=np.uint8)


def run_infer(ng_handle, ng_params_handle, in_vs, out_vs, in_name_, out_name_, arr):
    with hpf.InferVStreams(ng_handle, in_vs, out_vs) as infer_pipeline:
        with ng_handle.activate(ng_params_handle):
            results = infer_pipeline.infer({in_name_: arr})
    return results[out_name_]

# ─────────────────────────────────────────────────────────────
# 기존 3ch CarIa — 유지
# ─────────────────────────────────────────────────────────────
@app.route('/batch_extract_features', methods=['POST'])
def batch_extract_features():
    try:
        req = request.get_json()
        b64_list = req.get('images')
        if not isinstance(b64_list, list):
            return jsonify(error="Missing 'images' list"), 400
        features = []
        for b64 in b64_list:
            arr = preprocess_3ch_bgr_uint8(b64)
            feat = run_infer(ng, ng_params, in_vs_carla, out_vs_carla, in_name, out_name, arr)
            feat_u8 = np.asarray(feat).reshape(-1).astype(np.uint8)
            features.append(base64.b64encode(feat_u8.tobytes()).decode('ascii'))
        return jsonify(status='ok', features=features, dtype='uint8', length=int(feat_u8.size))
    except Exception:
        logging.exception('batch_extract_features failed')
        return jsonify(error="Internal server error"), 500

# ─────────────────────────────────────────────────────────────
# 기존 3ch Real — 유지
# ─────────────────────────────────────────────────────────────
@app.route('/batch_extract_features_real', methods=['POST'])
def batch_extract_features_real():
    try:
        req = request.get_json()
        b64_list = req.get('images')
        if not isinstance(b64_list, list):
            return jsonify(error="Missing 'images' list"), 400
        features = []
        for b64 in b64_list:
            arr = preprocess_3ch_bgr_uint8(b64)
            feat = run_infer(ng_real, ng_params_real, in_vs_real, out_vs_real, in_name_real, out_name_real, arr)
            feat_u8 = np.asarray(feat).reshape(-1).astype(np.uint8)
            features.append(base64.b64encode(feat_u8.tobytes()).decode('ascii'))
        return jsonify(status='ok', features=features, dtype='uint8', length=int(feat_u8.size))
    except Exception:
        logging.exception('batch_extract_features_real failed')
        return jsonify(error="Internal server error"), 500

# ─────────────────────────────────────────────────────────────
# ★신규 4ch Real — 추가
# ─────────────────────────────────────────────────────────────
@app.route('/batch_extract_features_real_4ch', methods=['POST'])
def batch_extract_features_real_4ch():
    try:
        req = request.get_json()
        b64_list = req.get('images')
        if not isinstance(b64_list, list):
            return jsonify(error="Missing 'images' list"), 400
        features = []
        for b64 in b64_list:
            arr = preprocess_4ch_bgr_uint8(b64)  # (1,4,224,224) UINT8
            feat = run_infer(ng_real4, ng_params_real4, in_vs_real4, out_vs_real4, in_name_real4, out_name_real4, arr)
            feat_u8 = np.asarray(feat).reshape(-1).astype(np.uint8)
            features.append(base64.b64encode(feat_u8.tobytes()).decode('ascii'))
        return jsonify(status='ok', features=features, dtype='uint8', length=int(feat_u8.size))
    except Exception:
        logging.exception('batch_extract_features_real_4ch failed')
        return jsonify(error="Internal server error"), 500

# ─────────────────────────────────────────────────────────────
# 상태/메타
# ─────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify(status='ok'), 200

@app.route('/meta', methods=['GET'])
def meta():
    try:
        def _probe(ngh, ngp, ivs, ovs, in_n, out_n, ch):
            dummy = np.zeros((1,ch,224,224), dtype=np.uint8)
            try:
                out = run_infer(ngh, ngp, ivs, ovs, in_n, out_n, dummy)
                out = np.asarray(out).reshape(-1)
                return {"ok": True, "length": int(out.size)}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # 🛠 여기서 레이아웃/출력 차이 테스트
        zero = np.zeros((1,4,224,224), np.uint8)        # NCHW
        chk  = np.zeros((224,224,4), np.uint8)
        chk[::2,::2,:] = 255
        chk_nchw = chk.transpose(2,0,1)[None,...]
        o1 = run_infer(ng_real4, ng_params_real4, in_vs_real4, out_vs_real4, in_name_real4, out_name_real4, zero).reshape(-1)
        o2 = run_infer(ng_real4, ng_params_real4, in_vs_real4, out_vs_real4, in_name_real4, out_name_real4, chk_nchw).reshape(-1)
        diff = int((o1 != o2).sum())

        info = {

        }
        return jsonify(status='ok', meta=info), 200
    except Exception as e:
        logging.exception('meta failed')
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    # 포트 동일(5000)
    app.run(host='0.0.0.0', port=5000)
