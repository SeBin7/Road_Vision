# train.py
import os

os.environ['IPEX_VERBOSE'] = '0'


import glob
import cv2
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from datasets.Dataset_4ch_aug import WindowedDataset
from models.Mobilenet_hailo_4ch import MobileNetFeatureExtractor #Mobilenetv3-small의 백본 초기 가중치 사용
from models.GRU_MLP_xpu import GRU_MLP_Classifier_XPU as GRU #dropout 20% 반영

import intel_extension_for_pytorch as ipex

os.environ['IPEX_VERBOSE'] = '0'

import logging
# IPEX 로거 이름 추적 (보통 'IPEX' 혹은 '_logger' 등)
logger = logging.getLogger("IPEX")
# INFO 레벨 이상의 메시지를 출력하지 않도록 WARNING으로 설정
logger.setLevel(logging.WARNING)




torch.set_num_threads(2)
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"사용 디바이스: {device}")




def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def collect_labeled_videos(root_dir):
    video_list, class_names = [], sorted(os.listdir(root_dir))
    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir): continue
        for vf in glob.glob(os.path.join(cls_dir, '*.mp4')):
            video_list.append((vf, cls))
    return video_list, class_names

def create_hailo_calibration(video_root, output_dir, total_samples=128):
    video_list, class_names = collect_labeled_videos(video_root)
    per_class = total_samples // len(class_names)
    class_videos = {}
    for vp, cls in video_list:
        class_videos.setdefault(cls, []).append(vp)
    input_dir = os.path.join(output_dir, 'input_0')
    os.makedirs(input_dir, exist_ok=True)
    idx = 0

    for cls, vids in class_videos.items():
        samples_per_video = max(1, per_class // len(vids))
        count = 0

        for vp in vids:
            if count >= per_class: break
            pipeline = (
                f"filesrc location={vp} ! "
                "qtdemux ! h264parse ! vaapih264dec ! "
                "videoconvert ! video/x-raw,format=RGB ! appsink"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total == 0: continue
            step, extracted = max(1, total//samples_per_video), 0
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            for f in range(0, total, step):
                if extracted>=samples_per_video or count>=per_class: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if not ret: continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224,224))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray,50,150)
                edges = cv2.resize(edges,(224,224))
                frame4 = np.concatenate([rgb, edges[:,:,None]],axis=2).astype(np.uint8)
                path = os.path.join(input_dir, f"sample_{idx:05d}.npy")
                np.save(path, frame4); os.chmod(path,0o644)
                idx += 1; extracted += 1; count += 1
            cap.release()
        print(f"  {cls}: {count}장 저장")
    os.chmod(input_dir,0o755); os.chmod(output_dir,0o755)
    print(f"✅ Hailo calibration set 생성: {idx}개 파일 in {input_dir}")
    return output_dir

class SequenceConsistentFlip:
    def __init__(self, p=0.5):
        self.p = p
        self.do_flip = None
    def __call__(self, frames):
        if self.do_flip is None:
            self.do_flip = (random.random() < self.p)
        return [F.hflip(f) for f in frames] if self.do_flip else frames
    def reset(self):
        self.do_flip = None

def train():
    video_root = '/home/kyj28/workspace/Road_Vision/Road_Vision_video/Road_Vision/Train_videos'
    calibration_dir = '/home/kyj28/workspace/Road_Vision/Road_Vision_video/Road_Vision/src_video/hailo_calibration'
    calib_dir = create_hailo_calibration(video_root, calibration_dir)
    calib_npy_path = os.path.join(calib_dir,'input_0')

    video_list, class_names = collect_labeled_videos(video_root)
    label_map = {n:i for i,n in enumerate(class_names)}
    print(f"총 비디오 수: {len(video_list)}, 클래스: {label_map}")

    seq_flip = SequenceConsistentFlip(p=0.5)
    train_tf = transforms.Compose([transforms.Resize((224,224)),
                                  transforms.ColorJitter(0.3,0.3,0.3,0.1),
                                  transforms.ToTensor()])
    
    # 검증용: 증강 없음
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])                              

    # train/val split
    N = len(video_list); vN = int(0.2*N)
    train_v, val_v = video_list[vN:], video_list[:vN]

    train_ds = WindowedDataset(train_v, seq_len=5, stride=3,
                               transform=train_tf,
                               class_map=label_map,
                               pre_sequence_transform=seq_flip)
    
    val_ds   = WindowedDataset(val_v,   seq_len=5, stride=3,
                               transform=val_tf,
                               class_map=label_map,
                               pre_sequence_transform=None)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=4, prefetch_factor=4,
                              collate_fn=safe_collate)
    
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,
                              num_workers=4, prefetch_factor=4,
                              collate_fn=safe_collate)
    

    cnn = MobileNetFeatureExtractor(feature_dim=128).to(device)
    cls = GRU(feature_dim=128, hidden_dim=64,
              num_classes=len(class_names)).to(device)
    
    # 백본(Feature Extractor) 동결: 사전학습된 가중치 유지
    for param in cnn.backbone.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(cnn.parameters())+list(cls.parameters()), lr=1e-4)

    # Validation Loss 기반으로 학습률 감소
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    best_val_acc, patience, counter = 0.0, 3, 0

    for epoch in range(1,6):
        seq_flip.reset()
        cnn.train()
        cls.train()
        total_loss, correct, total = 0,0,0
        for bidx, batch in enumerate(train_loader):
            if batch is None: continue
            windows, labels = batch
            b, t, c, h, w = windows.size()
            inp = windows.view(b*t, c, h, w).to(device)
            lbl = labels.to(device)
            # IPEX optimize per epoch
            cnn, optimizer = ipex.optimize(model=cnn, optimizer=optimizer,
                                           dtype=torch.float32, inplace=True)
            cls, optimizer = ipex.optimize(model=cls, optimizer=optimizer,
                                           dtype=torch.float32, inplace=True)
            feat = cnn(inp).view(b, t, -1)
            out  = cls(feat)

            loss = criterion(out, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*b
            preds = torch.argmax(out,dim=1)
            correct += (preds==lbl).sum().item()
            total   += b
            print(f"[E{epoch}] Train B{bidx}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={correct/total:.4f}")

        train_acc  = correct/total
        train_loss = total_loss/total
        print(f"[Epoch {epoch}] ▶ Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # Validation
        cnn.eval()
        cls.eval()
        val_loss, correct, total = 0,0,0
        with torch.no_grad():
            for vb, batch in enumerate(val_loader):
                if batch is None: continue
                windows, labels = batch
                b, t, c, h, w = windows.size()
                inp = windows.view(b*t, c, h, w).to(device)
                lbl = labels.to(device)
                feat = cnn(inp).view(b, t, -1)
                out  = cls(feat)
                loss = criterion(out, lbl)
                val_loss += loss.item()*b
                preds = torch.argmax(out,dim=1)
                correct += (preds==lbl).sum().item()
                total   += b
                print(f"[E{epoch}] 🔍 Val B{vb}/{len(val_loader)}: Loss={loss.item():.4f}")

        val_acc  = correct/total
        val_loss = val_loss/total
        print(f"[Epoch {epoch}] ✅ Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # 스케줄러 스텝: Validation Loss 기준
        scheduler.step(val_loss)
 

        # Early stopping & save
        if val_acc > best_val_acc:
            print("🔼 Val improved: save models & onnx")
            best_val_acc = val_acc
            counter = 0
            torch.save(cnn.state_dict(), "cnn_feature_extractor_4ch.pth")
            torch.save(cls.state_dict(), "gru_mlp_classifier_4ch.pth")
            torch.save(cnn, "cnn_feature_extractor_4ch.pt")
            torch.save(cls, "gru_mlp_classifier_4ch.pt")

            # ONNX export
            dummy = torch.randn(1,4,224,224).to(device)
            torch.onnx.export(cnn, dummy,
                              "cnn_feature_extractor_4ch.onnx",
                              input_names=["input"], output_names=["feature"],
                              opset_version=11)
            print("✅ ONNX 저장 완료")
            print("🎯 Calibration set:", calib_npy_path)
        else:
            counter += 1
            print(f"😕 No improvement ({counter}/{patience})")
            if counter >= patience:
                print("🛑 Early stopping")
                break

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    train()
