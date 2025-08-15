# gru_mlp_xpu.py ── GRU → unrolled 구현 + XPU 호환
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Tuple

# ──────────────────────────────────────────────────────────────────────────────
# 1. GRUCellXPU ── nn.Linear 6개로 GRU 단일 스텝 계산
# ──────────────────────────────────────────────────────────────────────────────
class GRUCellXPU(nn.Module):
    """
    PyTorch nn.GRUCell를 완전히 대체하는 XPU 호환 셀.
    - input_size  : CNN 등에서 들어오는 feature 차원
    - hidden_size : 내부 hidden 차원
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # 입력→게이트 선형 변환 3개
        self.x2z = nn.Linear(input_size,  hidden_size, bias=True)
        self.x2r = nn.Linear(input_size,  hidden_size, bias=True)
        self.x2n = nn.Linear(input_size,  hidden_size, bias=True)

        # hidden→게이트 선형 변환 3개( bias 불필요 )
        self.h2z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h2r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h2n = nn.Linear(hidden_size, hidden_size, bias=False)

        # Xavier 초기화 (원 코드와 동일)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args
        ─
        x_t   : (batch, input_size)
        h_prev: (batch, hidden_size)
        Returns
        ─
        h_t   : (batch, hidden_size)
        """
        z_t = torch.sigmoid(self.x2z(x_t) + self.h2z(h_prev))   # update gate
        r_t = torch.sigmoid(self.x2r(x_t) + self.h2r(h_prev))   # reset gate
        n_t = torch.tanh(   self.x2n(x_t) + self.h2n(r_t * h_prev))
        h_t = (1 - z_t) * h_prev + z_t * n_t
        return h_t


# ──────────────────────────────────────────────────────────────────────────────
# 2. GRUBlockXPU ── 다중 스텝 처리 + 배치퍼스트
# ──────────────────────────────────────────────────────────────────────────────
class GRUBlockXPU(nn.Module):
    """
    nn.GRU와 동일한 인터페이스 중 (output, h_n) 반환.
    - batch_first=True 전용 · 단방향 · num_layers=1 기본
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = GRUCellXPU(input_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,                     # (batch, seq_len, input_size)
        h0: torch.Tensor | None = None      # (batch, hidden_size)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        device          = x.device
        hidden_size     = self.cell.hidden_size

        if h0 is None:
            h_t = x.new_zeros(batch, hidden_size, device=device)
        else:
            h_t = h0

        outputs = []
        for t in range(seq_len):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(h_t.unsqueeze(1))        # (batch,1,H)

        y = torch.cat(outputs, dim=1)               # (batch,seq_len,H)
        h_n = h_t.unsqueeze(0)                      # (1,batch,H)
        return y, h_n


# ──────────────────────────────────────────────────────────────────────────────
# 3. 최종 GRU-MLP Classifier (XPU 버전)
# ──────────────────────────────────────────────────────────────────────────────
class GRU_MLP_Classifier_XPU(nn.Module):
    """
    기존 GRU_MLP_Classifier와 동일한 시그니처 유지
    (feature_dim, hidden_dim, num_layers, num_classes 전부 전달 허용)
    단, 현재 num_layers>1, bidirectional은 미구현.
    """
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim : int = 64,
        num_layers : int = 1,
        num_classes: int = 4
    ):
        super().__init__()
        if num_layers != 1:
            raise NotImplementedError("현재 예시는 num_layers = 1만 지원합니다.")
        self.gru = GRUBlockXPU(feature_dim, hidden_dim)   # ← custom GRU
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # MLP 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in self.gru.modules():
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        """
        Args : x (batch, seq_len, feature_dim)
        Returns: logits (batch, num_classes)
        """
        _, h_n = self.gru(x)          # h_n (1,batch,H)
        logits = self.classifier(h_n[-1])
        return logits
