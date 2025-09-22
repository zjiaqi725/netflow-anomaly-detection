import torch, torch.nn as nn, torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse

# ========== args ==========
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test"], required=True,
                    help="train for training, test for evaluation")
args = parser.parse_args()

# ========== load & preprocess ==========
df = pd.read_csv("flows.csv")
features = ["packets","bytes","duration","avg_iat","std_iat","pps"]
X = df[features].fillna(0).values
X = np.log1p(X)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# split into train / test
X_train, X_test = train_test_split(Xs, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# ========== model ==========
class AE(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, n)
        )
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

device = torch.device("cpu")
model = AE(Xs.shape[1]).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
lossfn = nn.MSELoss()

# ========== training ==========
if args.mode == "train":
    loader = torch.utils.data.DataLoader(X_train_tensor, batch_size=128, shuffle=True)
    for epoch in range(50):
        total=0
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            out = model(b)
            loss = lossfn(out, b)
            loss.backward()
            opt.step()
            total += loss.item()*b.size(0)
        print(f"Epoch {epoch} loss {total/len(X_train_tensor):.6f}")
    torch.save(model.state_dict(), "ae_model.pth")

# ========== testing ==========
elif args.mode == "test":
    model.load_state_dict(torch.load("ae_model.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        recon = model(X_test_tensor)
        mse = torch.mean((recon - X_test_tensor)**2, dim=1).numpy()
    test_df = df.iloc[X_test_tensor.shape[0]*-1:].copy()  # 对应测试集部分
    test_df["recon_error"] = mse
    test_df.sort_values("recon_error", ascending=False).head(50).to_csv("ae_top_anom.csv", index=False)
    print("Test complete. Results saved to ae_top_anom.csv")
