import cv2
import torch
from PIL import Image
from collections import deque
from torchvision import transforms
from feature_extractor import CNNFeatureExtractor
from rnn_classifier import GRU_MLP_Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {0: "ice_road", 1: "wet_road", 2: "normal_road", 3: "broken"}

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    cnn = CNNFeatureExtractor().to(device)
    classifier = GRU_MLP_Classifier().to(device)
    cnn.load_state_dict(torch.load("cnn_feature_extractor.pth", map_location=device))
    classifier.load_state_dict(torch.load("gru_mlp_classifier.pth", map_location=device))
    cnn.eval()
    classifier.eval()

    cap = cv2.VideoCapture(0)  # 0: 기본 웹캠

    buffer = deque(maxlen=5)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffer.append(transform(img))

        if len(buffer) == 5:
            with torch.no_grad():
                x = torch.stack(list(buffer)).unsqueeze(0).to(device)  # (1, T, C, H, W)
                b, t, c, h, w = x.size()
                features = cnn(x.view(b * t, c, h, w))
                features = features.view(b, t, -1)
                logits = classifier(features)
                pred = torch.argmax(logits, dim=1)
                label = label_map[pred.item()]
                # 화면에 라벨 표시
                cv2.putText(frame, f"Prediction: {label}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 2)

        # 프레임 출력
        cv2.imshow("Live Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()