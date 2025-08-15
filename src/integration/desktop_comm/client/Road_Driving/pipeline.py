# pipeline.py
import os
import cv2
import time
import json
import base64
import requests
import threading
import queue
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F

from models.GRU_MLP_xpu import GRU_MLP_Classifier_XPU as GRU
from ui_utils import draw_info_panel, draw_prob_bars

class RoadVisionPipeline:
    """Server-CNN과 Local-GRU를 결합한 도로 상태 분류 파이프라인 클래스."""

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.classifier = self._load_classifier()

        self.cap_q = queue.Queue(maxsize=self.config['CAP_QUEUE'])
        self.disp_q = queue.Queue(maxsize=self.config['CAP_QUEUE'])
        self.enc_q = queue.Queue(maxsize=self.config['ENC_QUEUE'])
        self.feat_q = queue.Queue()
        
        self.STOP_SIGNAL = object()
        self.cap = None
        self.session = requests.Session()

    def _get_device(self):
        """사용 가능한 최적의 PyTorch 디바이스를 설정합니다."""
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            print("Using XPU device.")
            return torch.device('xpu')
        if torch.cuda.is_available():
            print("Using CUDA device.")
            return torch.device('cuda')
        print("Using CPU device.")
        return torch.device('cpu')

    def _load_classifier(self):
        """GRU 분류기 모델을 로드하고 평가 모드로 설정합니다."""
        cls = GRU(feature_dim=self.config['FEATURE_DIM'], num_classes=self.config['NUM_CLASSES']).to(self.device)
        state = torch.load(self.config['CLS_WEIGHT'], map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if "classifier.2.weight" in state and "classifier.3.weight" not in state:
            state["classifier.3.weight"] = state.pop("classifier.2.weight")
            state["classifier.3.bias"] = state.pop("classifier.2.bias", state.get("classifier.3.bias"))
        
        missing, unexpected = cls.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[Model Load] Missing: {missing}, Unexpected: {unexpected}")
        cls.eval()
        return cls

    def _bgr_to_jpeg_b64(self, img_bgr):
        """BGR 이미지를 리사이즈, JPEG 인코딩, Base64 인코딩합니다."""
        img = cv2.resize(img_bgr, self.config['TARGET_HW'], interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.config['JPEG_QUALITY']])
        if not ok: raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf).decode("ascii")

    def _decode_features(self, b64_list):
        """Base64로 인코딩된 특징 벡터를 float32 Numpy 배열로 디코딩합니다."""
        out = np.empty((len(b64_list), self.config['FEATURE_DIM']), dtype=np.float32)
        for i, s in enumerate(b64_list):
            raw = base64.b64decode(s)
            u8 = np.frombuffer(raw, dtype=np.uint8, count=self.config['FEATURE_DIM'])
            if self.config['DEQUANT_MODE'] == "unit":
                out[i, :] = u8.astype(np.float32) / 255.0
            else:
                out[i, :] = (u8.astype(np.float32) - 128.0) / 128.0
        return out

    def _post_batch(self, b64_list):
        """인코딩된 이미지 배치를 서버로 전송하고 특징 벡터를 받습니다."""
        payload = {"images": b64_list}
        r = self.session.post(self.config['SERVER_URL'], data=json.dumps(payload),
                              headers={"Content-Type": "application/json"}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"Server error: {data}")
        return data["features"]

    def _capture_thread(self, video_path):
        """[스레드 1] 비디오에서 프레임을 읽어 큐에 넣습니다."""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_read_period = (1.0 / max(fps, 1e-6)) if self.config['SYNC_TO_SRC_FPS'] else 0.0
        next_read_t = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    if self.config['LOOP_PLAYBACK']:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        next_read_t = time.time()
                        continue
                    else:
                        break
                
                try:
                    if self.disp_q.full(): self.disp_q.get_nowait()
                    self.disp_q.put_nowait(frame)
                    
                    if self.cap_q.full(): self.cap_q.get_nowait()
                    self.cap_q.put_nowait(frame)
                except queue.Full:
                    pass

                if src_read_period > 0:
                    delay = next_read_t + src_read_period - time.time()
                    if delay > 0: time.sleep(delay)
                    next_read_t = time.time()
        finally:
            self.cap_q.put(self.STOP_SIGNAL)
            self.disp_q.put(self.STOP_SIGNAL)
            print("Capture thread finished.")

    def _encode_thread(self):
        """[스레드 2] 캡처된 프레임을 인코딩하여 큐에 넣습니다."""
        try:
            while True:
                frame = self.cap_q.get()
                if frame is self.STOP_SIGNAL: break
                try:
                    b64 = self._bgr_to_jpeg_b64(frame)
                    self.enc_q.put(b64)
                except Exception as e:
                    print(f"[ENCODE] Error: {e}")
        finally:
            self.enc_q.put(self.STOP_SIGNAL)
            print("Encode thread finished.")

    def _network_thread(self):
        """[스레드 3] 인코딩된 데이터를 배치로 서버에 전송하고 결과를 큐에 넣습니다."""
        batch = []
        try:
            while True:
                item = self.enc_q.get()
                if item is self.STOP_SIGNAL:
                    if batch:
                        try:
                            feat_b64 = self._post_batch(batch)
                            self.feat_q.put(self._decode_features(feat_b64))
                        except Exception as e:
                            print(f"[NET] Flush failed: {e}")
                    break
                
                batch.append(item)
                if len(batch) >= self.config['BATCH']:
                    try:
                        feat_b64 = self._post_batch(batch)
                        self.feat_q.put(self._decode_features(feat_b64))
                    except Exception as e:
                        print(f"[NET] Batch failed: {e}")
                    finally:
                        batch = []
        finally:
            self.feat_q.put(self.STOP_SIGNAL)
            print("Network thread finished.")

    def _display_and_inference_loop(self):
        """[메인 스레드] 화면 출력, GRU 추론 및 사용자 입력을 처리합니다."""
        win_name = "Road-Vision (Modularized)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        
        try:
            tx, ty, tw, th = self.config['DISPLAY_GEOMS'][self.config['TARGET_DISPLAY']]
        except IndexError:
            tx, ty, tw, th = (0, 0, self.config['DISP_SIZE'][0], self.config['DISP_SIZE'][1])
        
        cv2.moveWindow(win_name, tx, ty)
        cv2.resizeWindow(win_name, tw, th)
        if self.config['FULLSCREEN']:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        seq_buffer = deque(maxlen=self.config['SEQ_LEN'])
        pending_feats = []
        last_label, last_conf, prob = None, None, None
        
        disp_fps_t0, disp_fps_cnt = time.time(), 0
        disp_fps = 0.0
        
        src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_wh = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        play_start_t = time.time()
        shown_frames = 0
        frame_idx = 0

        while True:
            while not self.feat_q.empty():
                try:
                    item = self.feat_q.get_nowait()
                    if item is self.STOP_SIGNAL:
                        pending_feats.append(self.STOP_SIGNAL)
                        break
                    pending_feats.extend([item[i] for i in range(item.shape[0])])
                except queue.Empty:
                    break

            if any(feat is self.STOP_SIGNAL for feat in pending_feats):
                break

            try:
                frame = self.disp_q.get(timeout=1.0)
            except queue.Empty:
                print("Display queue is empty. Assuming processing has ended.")
                break

            if frame is self.STOP_SIGNAL: break
            frame_idx += 1
            
            if pending_feats:
                feat = pending_feats.pop(0)
                seq_buffer.append(feat)
                if len(seq_buffer) == self.config['SEQ_LEN']:
                    x = torch.from_numpy(np.stack(seq_buffer)[None, ...]).to(self.device)
                    with torch.no_grad():
                        logit = self.classifier(x)
                        prob = F.softmax(logit, dim=1)[0]
                        idx = int(torch.argmax(prob).item())
                        last_conf = float(prob[idx].item() * 100.0)
                        last_label = self.config['LABEL_MAP'][idx]

            disp_fps_cnt += 1
            now = time.time()
            if now - disp_fps_t0 >= 0.5:
                disp_fps = disp_fps_cnt / (now - disp_fps_t0)
                disp_fps_t0, disp_fps_cnt = now, 0
            
            disp_frame = cv2.resize(frame, self.config['DISP_SIZE'], interpolation=cv2.INTER_AREA)
            
            draw_info_panel(disp_frame, fps=disp_fps, src=f"{src_wh[0]}x{src_wh[1]}@{src_fps:.1f}",
                            batch=self.config['BATCH'], jpeg_q=self.config['JPEG_QUALITY'],
                            label=last_label, conf=last_conf, server_url=self.config['SERVER_URL'],
                            q_sizes=f"{self.cap_q.qsize()}/{self.enc_q.qsize()}/{self.feat_q.qsize()}")
            if prob is not None:
                draw_prob_bars(disp_frame, prob.detach().cpu().numpy(), 
                               [self.config['LABEL_MAP'][i] for i in range(self.config['NUM_CLASSES'])])
            
            if self.config['SYNC_TO_SRC_FPS'] and src_fps > 0:
                target_t = play_start_t + (shown_frames / src_fps)
                delay = target_t - time.time()
                if delay > 0: time.sleep(delay)
            shown_frames += 1

            cv2.imshow(win_name, disp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: break
            elif key == ord('f'):
                is_fullscreen = cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, 
                                      cv2.WINDOW_NORMAL if is_fullscreen else cv2.WINDOW_FULLSCREEN)
                cv2.moveWindow(win_name, tx, ty)
                if is_fullscreen: cv2.resizeWindow(win_name, tw, th)

            if self.config['LOOP_PLAYBACK'] and self.cap.get(cv2.CAP_PROP_POS_FRAMES) < 1:
                play_start_t = time.time()
                shown_frames = 0
    
    def _cleanup(self):
        """사용한 모든 자원을 해제합니다."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.session.close()
        print("✅ Pipeline finished and resources cleaned up.")

    def run(self, video_path):
        """전체 비전 파이프라인을 실행합니다."""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cv2.setNumThreads(0)
        
        threads = [
            threading.Thread(target=self._capture_thread, args=(video_path,), daemon=True),
            threading.Thread(target=self._encode_thread, daemon=True),
            threading.Thread(target=self._network_thread, daemon=True)
        ]
        for t in threads:
            t.start()
        
        try:
            self._display_and_inference_loop()
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            self._cleanup()
