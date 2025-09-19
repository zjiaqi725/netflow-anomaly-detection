import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv("flows.csv")
# 选择数值特征
features = ["packets","bytes","duration","avg_iat","std_iat","pps"]
X = df[features].fillna(0).values
scaler = StandardScaler()
Xs = scaler.fit_transform(np.log1p(X))  # 推荐先 log1p

clf = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
clf.fit(Xs)
scores = -clf.decision_function(Xs)  # 越大越异常
df["anomaly_score"] = scores


df = df.sort_values("anomaly_score", ascending=False)
df.head(30).to_csv("top_anomalous_flows.csv", index=False)
print("Top anomalies saved to top_anomalous_flows.csv")

# 