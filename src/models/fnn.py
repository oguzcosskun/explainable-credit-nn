import torch
import torch.nn as nn


class FNN(nn.Module):
    """
    Feedforward Neural Network with BatchNorm — credit risk baseline.
    Architecture: Input -> 64 (BN, ReLU, Dropout) -> 32 (BN, ReLU, Dropout) -> 1
    Uses BCEWithLogitsLoss (no Sigmoid in forward — more numerically stable).
    """

    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.4):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        # Son katman: sigmoid YOK — BCEWithLogitsLoss ile kullanılacak
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        """Sigmoid uygulayarak 0-1 arası olasılık döndür."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


# === TEST ===
if __name__ == "__main__":
    model = FNN(input_dim=48)
    dummy = torch.randn(5, 48)
    logits = model(dummy)
    probas = model.predict_proba(dummy)
    print(f"Model: {model}")
    print(f"Input shape:  {dummy.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probas shape: {probas.shape}")
    print(f"Probas: {probas.detach().numpy().flatten()}")
    print("FNN with BatchNorm OK")