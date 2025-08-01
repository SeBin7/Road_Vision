from flask import Flask, request, jsonify
import cv2
import numpy as np
import hailo_platform as hpf
import base64

# Flask 애플리케이션 생성
app = Flask(__name__)

# 사용할 HEF(Hailo Executable Format) 파일 경로
hef_path = './second_cnn_feature_extractor.hef'

# ─────────────────────────────────────────────────────────────
# Hailo 디바이스 및 네트워크 그룹 초기화
# ─────────────────────────────────────────────────────────────
hef = hpf.HEF(hef_path)  # HEF 파일 로드
device = hpf.VDevice()   # 가상 디바이스 생성
configure_params = hpf.ConfigureParams.create_from_hef(hef, hpf.HailoStreamInterface.PCIe)

# HEF를 기반으로 네트워크 그룹 구성 (리스트로 반환되므로 [0] 사용)
network_group = device.configure(hef, configure_params)[0]

# 추론 전에 반드시 network_group을 활성화
network_group.activate()

# 이 활성화 상태를 기반으로 활성화 파라미터 저장 (반복 재활용을 위해)
network_group_params = network_group.create_params()

# 입력/출력 스트림 정보 추출 (보통 하나만 사용하므로 [0])
input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_info = hef.get_output_vstream_infos()[0]

# 입력/출력 스트림 파라미터 생성 (양자화 활성화 및 UINT8 포맷 지정)
input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
    network_group, quantized=True, format_type=hpf.FormatType.UINT8
)
output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
    network_group, quantized=True, format_type=hpf.FormatType.UINT8
)

# ─────────────────────────────────────────────────────────────
# 추론 API 엔드포인트 정의 (/infer)
# ─────────────────────────────────────────────────────────────
@app.route('/infer', methods=['POST'])
def receive_frame():
    try:
        # JSON에서 base64 인코딩된 프레임 수신
        data = request.get_json()
        if not data or 'frame_base64' not in data:
            return jsonify({'error': 'frame_base64 missing'}), 400

        b64frame = data['frame_base64']
        frame_idx = data.get('frame_idx', -1)

        # base64 → numpy buffer → OpenCV 이미지 디코딩 (BGR)
        img_data = base64.b64decode(b64frame)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("❌ OpenCV 이미지 디코딩 실패")
            return jsonify({'error': 'OpenCV decode failed'}), 400

        print("✅ 이미지 디코딩 성공")
        print(f"  • 원본 frame shape: {frame.shape}, dtype: {frame.dtype}")

        # BGR → RGB 변환 및 224x224 리사이즈
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))

        # UINT8 타입 유지, 배치 차원 추가, 메모리 정렬 보장
        frame_input = frame_resized.astype(np.uint8)
        frame_input = np.expand_dims(frame_input, axis=0)   # (1, 224, 224, 3)
        frame_input = np.ascontiguousarray(frame_input)

        print(f"🧪 전처리 완료 frame_input shape: {frame_input.shape}, dtype: {frame_input.dtype}")
        print(f"📥 Hailo input stream shape: {input_vstream_info.shape}")
        print(f"📥 Hailo input stream format type: {input_vstream_info.format.type}")

        # ─────────────────────────────────────────────────────
        # Hailo 추론 실행
        # ─────────────────────────────────────────────────────
        with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            # 반드시 활성화 context 내부에서 추론해야 함
            with network_group.activate(network_group_params):
                results = infer_pipeline.infer({input_vstream_info.name: frame_input})
                feature_vector = results[output_vstream_info.name][0]  # (128,) 벡터

        print(f"✅ 추론 성공! feature_vector shape: {feature_vector.shape}, dtype: {feature_vector.dtype}")

        # 결과 벡터를 base64로 인코딩하여 JSON 응답
        feature_b64 = base64.b64encode(feature_vector.tobytes()).decode('utf-8')

        return jsonify({
            'status': 'ok',
            'feature_vector_base64': feature_b64,
            'frame_idx': frame_idx
        })

    except Exception as e:
        print(f"[ERROR] /infer processing error: {e}")
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────
# 애플리케이션 실행 (외부 접속 허용 및 포트 8001 사용)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001)
