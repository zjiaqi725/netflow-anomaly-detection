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
├── flow_features.py   # 从 pcapng 提取流特征，生成 flows.csv
├── detect_isolation.py      # IsolationForest 异常检测
├── ae_detect.py             # Autoencoder 异常检测
├── detect_lof.py            # Local Outlier Factor 异常检测
├── detect_ocsvm.py          # One-Class SVM 异常检测
├── main.py                  # 一键运行四种方法 (Python)
├── requirements.txt         # 项目依赖
└── README.md
```

---

## ⚙️ 环境准备
1. 创建虚拟环境
```bash
conda create --name zz-ad python=3.8
conda activate zz-ad
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

---

## 🚀 使用方法

### Step 1: 准备数据
将待分析的 `.pcapng` 文件（如 `scp.pcapng`）放到项目目录下。

### Step 2: 特征抽取
```bash
python flow_features.py
```

运行后会生成：

`flows.csv`：每行对应一个网络流，包含统计特征（包数、字节数、持续时间、IAT 等）。

### Step 3: 运行单个检测方法
```bash
python detect_isolation.py    # IsolationForest
python ae_detect.py           # Autoencoder
python detect_lof.py          # Local Outlier Factor
python detect_ocsvm.py        # One-Class SVM
```

每个脚本运行后会生成一个结果文件，例如：

* top_anomalous_flows.csv
* ae_top_anom.csv
* lof_top_anom.csv
* ocsvm_top_anom.csv

### Step 4: 一键运行

如果希望顺序运行所有方法，可使用以下方式：
```bash
python main.py scp.pcapng
```

---

## 📊 结果解读

- 各结果文件包含：
  - `*_score`：异常分数，数值越大越可疑。  
  - 若 `flows.csv` 中存在 `label` 列（0=正常，1=异常），会额外输出：
    - **AUROC**、**AUPR**、**F1**、**Precision**、**Recall**  
    - `pred_bestF1`：基于 PR 曲线最佳阈值的预测结果  
    - `pred_topP`：基于 contamination 先验比例的预测结果  

- 输出文件示例：
  - `top_anomalous_flows.csv` （IsolationForest）
  - `ae_top_anom.csv` （Autoencoder）
  - `lof_top_anom.csv` （Local Outlier Factor）
  - `ocsvm_top_anom.csv` （One-Class SVM）

- Top-N 排序的流最可能是异常流，可结合 **Wireshark** 打开对应流量进行进一步分析。 
