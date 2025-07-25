import torch.nn as nn

class GRU_MLP_Classifier(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=64, num_layers=1, num_classes=4):
        super().__init__()
        # GRU (Gated Recurrent Unit) 레이어 정의
        # feature_dim : 입력 feature 차원 (예, CNN feature vector 크기)
        # hidden_dim  : GRU 내부 hidden state 차원
        # num_layers  : GRU 레이어 수 (기본 1)
        # batch_first : (batch, seq, feature) 순서 사용
        self.gru = nn.GRU(feature_dim, hidden_dim, num_layers, batch_first=True)

        # MLP(다층퍼셉트론) 분류기 정의:
        # GRU의 마지막 hidden state를 받아 2단의 fully-connected layer 거쳐 클래스 예측
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),   # hidden_dim → 64 노드
            nn.ReLU(),                   # 비선형 활성화
            nn.Linear(64, num_classes)   # 64 → 클래스 개수(예: 4클래스)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, feature_dim)
        # 예: (batch_size, 10, 128)
        
        _, h_n = self.gru(x)
        # h_n shape: (num_layers, batch, hidden_dim)
        # h_n[-1]: 가장 마지막 레이어의 마지막 time step의 hidden state 선택
        #          (batch, hidden_dim)

        out = self.classifier(h_n[-1])
        # out shape: (batch, num_classes)
        # 최종 MLP를 통해 각 클래스별 로짓(logit, 예측값) 산출

        return out
        # 보통 CrossEntropyLoss에서 사용 (softmax 적용 전)