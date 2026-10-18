"""Genera un archivo Parquet SINTÉTICO con la misma estructura del tráfico de CIC-YNU-IoTMal 2026.

IMPORTANTE: estos datos son ficticios y sirven SOLO para comprobar que los scripts se ejecutan de
principio a fin (prueba de humo). No se usan en el artículo ni producen resultados reportables.

Uso:
  python tests/generar_datos_prueba.py --destino data/raw/prueba
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNAS = ["flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
            "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
            "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
            "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP",
            "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "IAT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default="data/raw/prueba")
    ap.add_argument("--binarios", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    familias = {"Mirai": 1.0, "DarkNexus": 1.6, "Gafgyt": 2.2, "Benign": 3.0, "Unknown": 1.3}
    filas = []
    for fam, desp in familias.items():
        for b in range(args.binarios):
            sha = f"{fam[:3].lower()}{b:04d}" + "0" * 56
            arch = rng.choice(["ARM", "MIPS", "MIPSEL", "x86"])
            n = int(rng.integers(20, 60))
            X = rng.gamma(2.0, 1.0 + desp, size=(n, len(COLUMNAS)))
            X[:, 22] = rng.binomial(1, 0.15 * desp, n)          # Telnet
            X[:, 8] = rng.binomial(1, min(0.9, 0.2 * desp), n)  # syn_flag_number
            X[:, -1] = rng.uniform(0, 1e5, n)                    # IAT (atajo)
            df = pd.DataFrame(X, columns=COLUMNAS)
            df["label"] = fam if fam != "Gafgyt" else "bashlite"
            df["sha256"] = sha
            df["arch"] = arch
            filas.append(df)
    out = pd.concat(filas, ignore_index=True)
    out.loc[out.sample(frac=0.002, random_state=1).index, "Rate"] = np.inf
    d = Path(args.destino)
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / "prueba_pcap.parquet", index=False)
    print(f"Archivo de prueba: {d / 'prueba_pcap.parquet'} ({len(out)} filas)")


if __name__ == "__main__":
    main()
