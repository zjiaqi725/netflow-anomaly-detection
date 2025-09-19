# Network Flow Anomaly Detection

本项目提供了一个 **多维度的网络流量异常检测工具集**，支持从 `.pcapng` 抓包文件中自动提取网络流特征，并基于多种无监督方法进行检测。  
主要应用于：入侵检测、数据外泄监控、异常流量分析等。

---

## ✨ 功能特点
- **流特征抽取**：基于 Scapy，直接解析 `.pcapng`，无需安装 `tshark`。
- **多模型检测**：集成四种无监督算法，覆盖不同维度：
  - IsolationForest：稀有性，全局孤立点检测  
  - Autoencoder (AE)：重构误差，模式偏离检测  
  - Local Outlier Factor (LOF)：局部密度差异检测  
  - One-Class SVM (OCSVM)：学习正常边界，识别域外点  
- **一键运行**：main.py，可顺序运行所有方法。
- **可评估指标**：若在 `flows.csv` 中提供 `label` 列（0=正常，1=异常），会输出 AUROC、AUPR、F1、Precision、Recall 等。

---

## 📂 项目结构

```text
.
├── flow_features_scapy.py   # 从 pcapng 提取流特征，生成 flows.csv
├── detect_isolation.py      # IsolationForest 异常检测
├── ae_detect.py             # Autoencoder 异常检测
├── detect_lof.py            # Local Outlier Factor 异常检测
├── detect_ocsvm.py          # One-Class SVM 异常检测
├── main.py                  # 一键运行四种方法 (Python)
├── requirements.txt         # 项目依赖
└── README.md
