import torch
from feature_extractor import CNNFeatureExtractor
from rnn_classifier import GRU_MLP_Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(cnn_path, rnn_path):
    cnn = CNNFeatureExtractor(feature_dim=128).to(device)
    classifier = GRU_MLP_Classifier(feature_dim=128, hidden_dim=64, num_classes=4).to(device)

    cnn.load_state_dict(torch.load(cnn_path))
    classifier.load_state_dict(torch.load(rnn_path))

    cnn.eval()
    classifier.eval()
    return cnn, classifier

def infer_window(cnn, classifier, window_tensor):
    with torch.no_grad():
        feats = cnn(window_tensor.to(device))
        feats = feats.unsqueeze(0)  # (1, window_size, feature_dim)
        preds = classifier(feats)
        probs = torch.softmax(preds, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        return pred_label, probs.cpu().numpy()

if __name__ == "__main__":
    # 예시: test용 입력 데이터를 별도로 준비해서 호출 필요
    pass
