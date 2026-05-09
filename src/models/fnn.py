import torch
import torch.nn as nn

class FNN(nn.Module):
    """
    Basit Feedforward Neural Network — credit risk baseline.
    
    Mimari:
        Input → 64 (ReLU, Dropout) → 32 (ReLU, Dropout) → 1 (Sigmoid)
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        
        # Son katman: tek nöron, sigmoid ile olasılık çıktısı
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# === DOĞRUDAN ÇALIŞTIRMA TESTİ ===
if __name__ == "__main__":
    # 19 feature'lık dummy input ile test
    model = FNN(input_dim=19)
    dummy_input = torch.randn(5, 19)  # 5 örnek, 19 feature
    output = model(dummy_input)
    print(f"Model: {model}")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.detach().numpy().flatten()}")
    print("FNN model OK ✅")