"""Paso 1. Exploración del dataset crudo (sin modificarlo).

Referencia en el paper: Sección 3.2 (Descripción del dataset, Tabla 2).

Lee los archivos de tráfico de red de data/raw (Parquet o CSV) y registra:
  - archivos, filas y columnas por archivo;
  - columna de etiqueta, de binario (hash) y de arquitectura detectadas;
  - distribución de ventanas y de binarios por familia y por arquitectura.
Salidas: results/exploracion/resumen_exploracion.json y results/exploracion/resumen_exploracion.md
"""
import pandas as pd

from common import (arquitectura_desde_nombre, cargar_config, detectar_columna_arquitectura,
                    detectar_columna_etiqueta, detectar_columna_grupo, guardar_json, listar_archivos,
                    logger, normalizar_familia, ruta)

log = logger("01_exploracion")


def leer_columnas(path, columnas=None):
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columnas)
    return pd.read_csv(path, usecols=columnas, low_memory=False)


def esquema(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return pf.schema_arrow.names, pf.metadata.num_rows
    cols = pd.read_csv(path, nrows=5).columns.tolist()
    with open(path, "rb") as f:
        n = sum(1 for _ in f) - 1
    return cols, n


def main():
    cfg = cargar_config()
    archivos = listar_archivos(cfg)
    if not archivos:
        log.error("No se encontraron archivos en %s. Ejecute primero 00_descargar_datos.py", cfg["datos"]["dir_raw"])
        return
    resumen = {"archivos": [], "familias_ventanas": {}, "familias_binarios": {}, "arquitecturas": {}}
    dist_total = []
    for p in archivos:
        cols, n = esquema(p)
        col_lab = detectar_columna_etiqueta(cols, cfg)
        col_grp = detectar_columna_grupo(cols, cfg)
        col_arq = detectar_columna_arquitectura(cols, cfg)
        info = {"archivo": str(p.relative_to(ruta(cfg["datos"]["dir_raw"]))), "filas": n, "n_columnas": len(cols),
                "columnas": cols, "columna_etiqueta": col_lab, "columna_grupo": col_grp, "columna_arquitectura": col_arq}
        resumen["archivos"].append(info)
        log.info("%s: %d filas, %d columnas | etiqueta=%s | grupo=%s | arq=%s", p.name, n, len(cols), col_lab, col_grp, col_arq)
        if col_lab is None:
            continue
        usar = [c for c in [col_lab, col_grp, col_arq] if c]
        df = leer_columnas(p, usar)
        df["familia"] = df[col_lab].map(lambda v: normalizar_familia(v, cfg))
        df["arquitectura"] = df[col_arq].astype(str) if col_arq else (arquitectura_desde_nombre(p.name) or "NA")
        df["grupo"] = df[col_grp].astype(str) if col_grp else "NA"
        dist_total.append(df[["familia", "arquitectura", "grupo"]])

    if dist_total:
        d = pd.concat(dist_total, ignore_index=True)
        resumen["familias_ventanas"] = d["familia"].value_counts().to_dict()
        resumen["familias_binarios"] = d.groupby("familia")["grupo"].nunique().to_dict()
        resumen["arquitecturas"] = d["arquitectura"].value_counts().to_dict()
        resumen["tabla_familia_arquitectura"] = (d.groupby(["familia", "arquitectura"]).size()
                                                 .unstack(fill_value=0).to_dict())
    guardar_json(resumen, "results/exploracion/resumen_exploracion.json")

    md = ["# Resumen de exploración del dataset", "", "| Archivo | Filas | Columnas | Etiqueta | Grupo | Arquitectura |",
          "|---|---:|---:|---|---|---|"]
    for a in resumen["archivos"]:
        md.append(f"| {a['archivo']} | {a['filas']} | {a['n_columnas']} | {a['columna_etiqueta']} | {a['columna_grupo']} | {a['columna_arquitectura']} |")
    md += ["", "## Ventanas y binarios por familia", "", "| Familia | Ventanas | Binarios |", "|---|---:|---:|"]
    for fam, n in resumen["familias_ventanas"].items():
        md.append(f"| {fam} | {n} | {resumen['familias_binarios'].get(fam, 'NA')} |")
    md += ["", "## Ventanas por arquitectura", "", "| Arquitectura | Ventanas |", "|---|---:|"]
    for a, n in resumen["arquitecturas"].items():
        md.append(f"| {a} | {n} |")
    destino = ruta("results/exploracion/resumen_exploracion.md")
    destino.write_text("\n".join(md), encoding="utf-8")
    log.info("Resumen guardado en %s", destino)


if __name__ == "__main__":
    main()
