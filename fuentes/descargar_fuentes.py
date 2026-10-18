"""Descarga los PDF de acceso abierto de las referencias del artículo en la carpeta fuentes/.

Referencia en el paper: sección "Referencias"; la correspondencia sección -> archivo está en docs/trazabilidad.md.

Lee fuentes/manifest_fuentes.csv (código, cita, archivo, url_pdf, doi_o_pagina, acceso). Para cada referencia
con url_pdf intenta la descarga y comprueba que el archivo sea un PDF. Las que no se pueden descargar
(suscripción, libro o bloqueo del sitio) se listan en fuentes/pendientes_descarga_manual.csv con su enlace,
para descargarlas manualmente (por ejemplo, desde la biblioteca de la universidad) con el nombre indicado.

Uso:
  python fuentes/descargar_fuentes.py
"""
import csv
import time
from pathlib import Path

import requests

DIR = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36", "Accept": "application/pdf,*/*"}


def main():
    pendientes = []
    with open(DIR / "manifest_fuentes.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    for r in filas:
        destino = DIR / r["archivo"]
        if destino.exists() and destino.stat().st_size > 10_000:
            print(f"[ya existe] {r['codigo']} {destino.name}")
            continue
        url = r["url_pdf"].strip()
        if not url:
            pendientes.append(r)
            print(f"[manual]    {r['codigo']} {r['cita']} -> {r['doi_o_pagina']}")
            continue
        try:
            resp = requests.get(url, headers=UA, timeout=60, allow_redirects=True)
            ok = resp.status_code == 200 and resp.content[:5] == b"%PDF-"
        except requests.RequestException:
            ok = False
        if ok:
            destino.write_bytes(resp.content)
            print(f"[descargado]{r['codigo']} {destino.name} ({len(resp.content) // 1024} KB)")
        else:
            pendientes.append(r)
            print(f"[manual]    {r['codigo']} {r['cita']} -> {r['doi_o_pagina'] or url}")
        time.sleep(1.0)
    with open(DIR / "pendientes_descarga_manual.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(pendientes)
    print(f"\nPendientes de descarga manual: {len(pendientes)} (ver fuentes/pendientes_descarga_manual.csv)")


if __name__ == "__main__":
    main()
