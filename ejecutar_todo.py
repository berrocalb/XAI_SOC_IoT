"""Ejecuta el experimento completo en orden (pasos 01 a 06) y registra el tiempo de cada paso.

Uso:
  python ejecutar_todo.py            # experimento completo
  python ejecutar_todo.py --rapido   # prueba rápida del paso 04 (pocas alertas)
  python ejecutar_todo.py --desde 3  # retoma desde el paso 3
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PASOS = ["01_explorar_dataset.py", "02_preprocesar.py", "03_entrenar_detector.py",
         "04_explicar_alertas.py", "05_analisis_estadistico.py", "06_figuras_tablas.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--desde", type=int, default=1)
    args = ap.parse_args()
    tiempos = []
    for i, paso in enumerate(PASOS, start=1):
        if i < args.desde:
            continue
        cmd = [sys.executable, str(RAIZ / "src" / paso)]
        if args.rapido and paso.startswith("04"):
            cmd.append("--rapido")
        print(f"\n=== Paso {i}: {paso} ===", flush=True)
        t0 = time.perf_counter()
        entorno = dict(os.environ, PYTHONWARNINGS="ignore")
        r = subprocess.run(cmd, cwd=RAIZ, env=entorno)
        dt = time.perf_counter() - t0
        tiempos.append((paso, dt))
        if r.returncode != 0:
            print(f"El paso {paso} terminó con error (código {r.returncode}).")
            sys.exit(r.returncode)
    print("\nResumen de tiempos:")
    for paso, dt in tiempos:
        print(f"  {paso:32s} {dt / 60:6.1f} min")


if __name__ == "__main__":
    main()
