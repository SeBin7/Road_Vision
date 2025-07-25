import torch
from torch.utils.data import DataLoader
from dataset import WindowedDataset
from feature_extractor import CNNFeatureExtractor
from rnn_classifier import GRU_MLP_Classifier
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # 각 시퀀스(폴더) 당 5장 프레임 샘플을 윈도우로 취급함
    dataset = WindowedDataset('../data/frames', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    cnn = CNNFeatureExtractor(feature_dim=128).to(device)
    classifier = GRU_MLP_Classifier(feature_dim=128, hidden_dim=64, num_classes=4).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(classifier.parameters()), lr=0.001)

    for epoch in range(5):
        cnn.train()
        classifier.train()
        total_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            b, win, c, h, w = batch_x.size()
            batch_x = batch_x.view(b * win, c, h, w).to(device)
            batch_y = batch_y.to(device)

            features = cnn(batch_x)
            features = features.view(b, win, -1)
            output = classifier(features)

            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * b

            # --- 상태 출력 ---
            print(f"[Epoch {epoch+1}/5] "
                f"Batch {batch_idx+1}/{len(dataloader)} "
                f"Loss: {loss.item():.4f}")

        mean_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}, Mean Loss: {mean_loss:.4f}\n")

    # === 학습 완료 후 모델 저장 ===
    torch.save(cnn.state_dict(), 'cnn_feature_extractor.pth')
    torch.save(classifier.state_dict(), 'gru_mlp_classifier.pth')
    print("모델 저장이 완료되었습니다.")

if __name__ == "__main__":
    train()
