"""Paso 0. Descarga del dataset CIC-YNU-IoTMal 2026 desde Kaggle.

Referencia en el paper: Sección 3.2 (Descripción del dataset) y sección "Dataset y código fuente".

Requisitos:
  1. Cuenta de Kaggle y token de API (Kaggle > Settings > API > "Create New Token").
  2. Guardar kaggle.json en ~/.kaggle/kaggle.json (Linux/macOS) o
     C:\\Users\\<usuario>\\.kaggle\\kaggle.json (Windows).

Uso:
  python src/00_descargar_datos.py                # usa el dataset definido en config.yaml
  python src/00_descargar_datos.py --dataset owner/slug

Si no es posible usar Kaggle, descargue el tráfico de red (PCAP procesado, Parquet) desde la página
oficial del CIC (https://www.unb.ca/cic/datasets/ynu-iot-2026.html) y copie los archivos en data/raw/.
"""
import argparse
import hashlib
import zipfile

from common import cargar_config, guardar_json, logger, ruta

log = logger("00_descarga")


def sha256(path, bloque=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bloque), b""):
            h.update(b)
    return h.hexdigest()


def main():
    cfg = cargar_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=cfg["kaggle"]["dataset"])
    args = ap.parse_args()
    destino = ruta(cfg["datos"]["dir_raw"])
    destino.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        log.error("Instale el paquete 'kaggle' (pip install kaggle).")
        return
    api = KaggleApi()
    api.authenticate()
    log.info("Descargando %s en %s ...", args.dataset, destino)
    api.dataset_download_files(args.dataset, path=str(destino), unzip=False, quiet=False)

    for z in destino.glob("*.zip"):
        log.info("Descomprimiendo %s", z.name)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(destino)

    # Registro de integridad (SHA-256) de todos los archivos descargados
    registro = []
    for p in sorted(destino.rglob("*")):
        if p.is_file() and p.name != ".gitkeep":
            registro.append({"archivo": str(p.relative_to(destino)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    guardar_json({"dataset": args.dataset, "archivos": registro}, "results/exploracion/integridad_descarga.json")
    log.info("Integridad registrada en results/exploracion/integridad_descarga.json (%d archivos)", len(registro))


if __name__ == "__main__":
    main()
