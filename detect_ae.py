import torch, torch.nn as nn, torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("flows.csv")
features = ["packets","bytes","duration","avg_iat","std_iat","pps"]
X = df[features].fillna(0).values
X = np.log1p(X)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

X_tensor = torch.tensor(Xs, dtype=torch.float32)

class AE(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, n))
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

device = torch.device("cpu")
model = AE(Xs.shape[1]).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
lossfn = nn.MSELoss()

loader = torch.utils.data.DataLoader(X_tensor, batch_size=128, shuffle=True)
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
    print(f"Epoch {epoch} loss {total/len(X_tensor):.6f}")

# compute reconstruction error
with torch.no_grad():
    recon = model(X_tensor)
    mse = torch.mean((recon - X_tensor)**2, dim=1).numpy()
df["recon_error"] = mse
df.sort_values("recon_error", ascending=False).head(50).to_csv("ae_top_anom.csv", index=False)
