# flow_features_scapy.py
import pandas as pd
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP

PCAP_FILE = "scp.pcapng"
OUTPUT_CSV = "flows.csv"

# 读取数据包
print(f"Reading {PCAP_FILE} ...")
packets = rdpcap(PCAP_FILE)
print(f"Total packets loaded: {len(packets)}")

flows = {}

def flow_key(pkt):
    """构造 flow 的 key: (src, sport, dst, dport, proto)"""
    if IP not in pkt:
        return None
    ip = pkt[IP]
    proto = None
    sport = None
    dport = None
    if TCP in pkt:
        proto = "TCP"
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
    elif UDP in pkt:
        proto = "UDP"
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    else:
        proto = str(ip.proto)  # 其他协议
    return (ip.src, sport, ip.dst, dport, proto)

for pkt in packets:
    k = flow_key(pkt)
    if not k:
        continue
    ts = float(pkt.time)
    length = len(pkt)

    f = flows.get(k)
    if f is None:
        flows[k] = {
            "src": k[0],
            "dst": k[2],
            "sport": k[1],
            "dport": k[3],
            "proto": k[4],
            "packets": 1,
            "bytes": length,
            "start": ts,
            "end": ts,
            "last_ts": ts,
            "iats": []
        }
    else:
        f["packets"] += 1
        f["bytes"] += length
        f["end"] = ts
        f["iats"].append(ts - f["last_ts"])
        f["last_ts"] = ts

# 聚合成 DataFrame
rows = []
for k, f in flows.items():
    dur = f["end"] - f["start"]
    iats = np.array(f["iats"])
    rows.append({
        "src": f["src"],
        "dst": f["dst"],
        "sport": f["sport"],
        "dport": f["dport"],
        "proto": f["proto"],
        "packets": f["packets"],
        "bytes": f["bytes"],
        "duration": dur,
        "avg_iat": iats.mean() if iats.size > 0 else 0.0,
        "std_iat": iats.std() if iats.size > 0 else 0.0,
        "min_iat": iats.min() if iats.size > 0 else 0.0,
        "max_iat": iats.max() if iats.size > 0 else 0.0,
        "pps": f["packets"]/dur if dur > 0 else f["packets"]
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {OUTPUT_CSV}, total flows={len(df)}")
