import os, subprocess, sys, shutil

PCAP = sys.argv[1] if len(sys.argv) > 1 else "scp.pcapng"
PY = sys.executable

def run(cmd):
    print("[$]", " ".join(cmd))
    subprocess.check_call(cmd)

def ensure_packages():
    print(">>> 检查/安装依赖")
    pkgs = ["pandas","numpy","scikit-learn","torch","scapy"]
    for p in pkgs:
        try:
            __import__(p if p!="scikit-learn" else "sklearn")
        except Exception:
            run([PY,"-m","pip","install","-q",p])
    print("依赖OK")

def ensure_flows():
    if not os.path.exists("flows.csv"):
        if not os.path.exists(PCAP):
            raise FileNotFoundError(f"未找到 {PCAP}，请放到当前目录或传参指定")
        print(f">>> 生成 flows.csv 自 {PCAP}")
        run([PY,"flow_features_scapy.py"])
    else:
        print(">>> 已存在 flows.csv，跳过特征抽取")

def run_detectors():
    print(">>> 运行 IsolationForest")
    run([PY,"detect_isolation.py"])
    print(">>> 运行 LOF")
    run([PY,"detect_lof.py"])
    print(">>> 运行 One-Class SVM")
    run([PY,"detect_ocsvm.py"])
    print(">>> 运行 Autoencoder")
    run([PY,"detect_ae.py"])

def summary():
    print(">>> 产物概览（若存在则列出）")
    for f in ["flows.csv",
              "top_anomalous_flows.csv",
              "lof_top_anom.csv",
              "ocsvm_top_anom.csv",
              "ae_top_anom.csv"]:
        if os.path.exists(f):
            sz = os.path.getsize(f)
            print(f" - {f}  ({sz/1024:.1f} KB)")
    print("完成 ✅")

if __name__ == "__main__":
    ensure_packages()
    ensure_flows()
    run_detectors()
    summary()
