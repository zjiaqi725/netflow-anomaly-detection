import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, confusion_matrix, precision_score, recall_score
)

# ==== 可配置 ====
INPUT_CSV = "flows.csv"
OUTPUT_TOPK = "ocsvm_top_anom.csv"
FEATURES = ["packets","bytes","duration","avg_iat","std_iat","pps"]
KERNEL = "rbf"        # 常用 rbf；也可 'linear', 'poly', 'sigmoid'
NU = 0.01             # 训练时的异常比例上界（0~1）；可与 contamination 同步
GAMMA = "scale"       # rbf带宽；可设 'auto' 或具体数值
CONTAMINATION = 0.01  # 仅用于top-p阈值策略
SEED = 42

def compute_supervised_metrics(y_true, scores, strategy="best_f1", contamination=0.01):
    metrics = {}
    order = np.argsort(-scores)
    s_sorted = scores[order]

    try:
        metrics["auroc"] = roc_auc_score(y_true, scores)
    except Exception:
        metrics["auroc"] = np.nan
    try:
        metrics["aupr"] = average_precision_score(y_true, scores)
    except Exception:
        metrics["aupr"] = np.nan

    if strategy == "top_p":
        k = max(1, int(np.ceil(contamination * len(scores))))
        thr = s_sorted[k-1] if k <= len(scores) else s_sorted[-1]
    else:
        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
        f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
        best_idx = np.nanargmax(f1s)
        thr = thresholds[best_idx]

    y_pred = (scores >= thr).astype(int)
    metrics["threshold"] = float(thr)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["cm"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics, y_pred

def main():
    df = pd.read_csv(INPUT_CSV)

    X = df[FEATURES].fillna(0).values
    X = np.log1p(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    oc = OneClassSVM(kernel=KERNEL, nu=NU, gamma=GAMMA)  # nu~异常率上界
    oc.fit(Xs)

    # decision_function：正值越“正常”；我们要“越大越异常”，因此反向
    raw = oc.decision_function(Xs)  # 通常负值为异常
    scores = -raw                   # 取相反数：越大越异常
    df["ocsvm_score"] = scores

    if "label" in df.columns:
        y_true = df["label"].astype(int).values
        print("Found 'label' column. Computing supervised metrics...")

        m_bestf1, y_pred_bestf1 = compute_supervised_metrics(y_true, scores, strategy="best_f1")
        print("[Best-F1] AUROC={:.4f} AUPR={:.4f} F1={:.4f} P={:.4f} R={:.4f} thr={:.6f}".format(
            m_bestf1["auroc"], m_bestf1["aupr"], m_bestf1["f1"], m_bestf1["precision"], m_bestf1["recall"], m_bestf1["threshold"]
        ))
        print("CM:", m_bestf1["cm"])
        df["pred_bestF1"] = y_pred_bestf1

        m_topp, y_pred_topp = compute_supervised_metrics(y_true, scores, strategy="top_p", contamination=CONTAMINATION)
        print("[Top-p ] AUROC={:.4f} AUPR={:.4f} F1={:.4f} P={:.4f} R={:.4f} thr={:.6f}".format(
            m_topp["auroc"], m_topp["aupr"], m_topp["f1"], m_topp["precision"], m_topp["recall"], m_topp["threshold"]
        ))
        print("CM:", m_topp["cm"])
        df["pred_topP"] = y_pred_topp
    else:
        print("No 'label' column — skipping AUROC/AUPR/F1.")

    df.sort_values("ocsvm_score", ascending=False).head(50).to_csv(OUTPUT_TOPK, index=False)
    print(f"Saved top OCSVM anomalies to {OUTPUT_TOPK}. Total={len(df)}")

if __name__ == "__main__":
    main()
