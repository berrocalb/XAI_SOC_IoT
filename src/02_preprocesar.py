"""Paso 2. Preprocesamiento del tráfico crudo.

Referencia en el paper: Sección 3.3 (Preprocesamiento).

Pasos (todos registrados en results/exploracion/reporte_preprocesamiento.json):
  a) Normalización de familias y selección de las familias incluidas (config: datos.clases_incluidas);
     se excluyen las ejecuciones benignas y las etiquetas Unknown/Generic/Other.
  b) Eliminación de identificadores y atajos de captura (config: preprocesamiento.excluir_*),
     columnas no numéricas y columnas constantes.
  c) Eliminación de valores infinitos o nulos y de vectores duplicados (características + familia).
  d) Submuestreo por binario (grupos completos) hasta max_ventanas_por_clase.
  e) Partición 70/30 agrupada por binario (hash SHA-256), estratificada por familia.
Salidas: data/processed/train.parquet, data/processed/test.parquet, data/processed/caracteristicas.json
"""
import numpy as np
import pandas as pd

from common import (arquitectura_desde_nombre, cargar_config, columnas_excluidas, detectar_columna_arquitectura,
                    detectar_columna_etiqueta, detectar_columna_grupo, fijar_semilla, guardar_json,
                    listar_archivos, logger, normalizar_familia, ruta)

log = logger("02_preproc")


def leer(path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def particion_agrupada(df, prop, rng, min_grupos=10):
    """Asigna cada binario (grupo) a train o test dentro de cada familia."""
    es_test = np.zeros(len(df), dtype=bool)
    detalle = {}
    agrupada = True
    for fam, sub in df.groupby("familia"):
        grupos = rng.permutation(np.asarray(sub["grupo"].unique(), dtype=object))
        if len(grupos) < min_grupos:
            agrupada = False
        n_test = max(1, int(round(prop * len(grupos))))
        g_test = set(grupos[:n_test])
        mask = df["familia"].eq(fam) & df["grupo"].isin(g_test)
        es_test |= mask.to_numpy()
        detalle[fam] = {"binarios_train": int(len(grupos) - n_test), "binarios_test": int(n_test)}
    return es_test, detalle, agrupada


def main():
    cfg = cargar_config()
    semilla = cfg["semilla"]
    fijar_semilla(semilla)
    rng = np.random.default_rng(semilla)
    incluidas = {c.lower(): c for c in cfg["datos"]["clases_incluidas"]}
    rep = {"pasos": []}

    partes = []
    for p in listar_archivos(cfg):
        df = leer(p)
        col_lab = detectar_columna_etiqueta(df.columns, cfg)
        if col_lab is None:
            log.warning("Sin columna de etiqueta en %s; se omite", p.name)
            continue
        col_grp = detectar_columna_grupo(df.columns, cfg)
        col_arq = detectar_columna_arquitectura(df.columns, cfg)
        n0 = len(df)
        fam = df[col_lab].map(lambda v: normalizar_familia(v, cfg)).astype(str)
        df["familia"] = fam.map(lambda f: incluidas.get(f.lower(), None))
        df = df[df["familia"].notna()].copy()
        df["grupo"] = df[col_grp].astype(str) if col_grp else p.stem
        df["arquitectura"] = df[col_arq].astype(str) if col_arq else (arquitectura_desde_nombre(p.name) or "NA")
        df["archivo_origen"] = p.name
        rep["pasos"].append({"archivo": p.name, "filas_originales": n0, "filas_familias_incluidas": len(df),
                             "columna_etiqueta": col_lab, "columna_grupo": col_grp})
        log.info("%s: %d -> %d ventanas de familias incluidas", p.name, n0, len(df))
        partes.append(df)
    if not partes:
        log.error("No hay datos que procesar.")
        return
    df = pd.concat(partes, ignore_index=True)
    meta = ["familia", "grupo", "arquitectura", "archivo_origen"]

    # b) selección de características
    excl = set(columnas_excluidas(df.columns, cfg))
    numericas = [c for c in df.columns if c not in meta and c not in excl and pd.api.types.is_numeric_dtype(df[c])]
    no_numericas = [c for c in df.columns if c not in meta and c not in excl and c not in numericas]
    X = df[numericas].astype("float32")
    constantes = [c for c in numericas if X[c].nunique(dropna=True) <= 1]
    feats = [c for c in numericas if c not in constantes]
    rep["columnas_excluidas_por_nombre"] = sorted(map(str, excl))
    rep["columnas_no_numericas"] = no_numericas
    rep["columnas_constantes"] = constantes
    df = pd.concat([X[feats], df[meta]], axis=1)

    # c) infinitos, nulos y duplicados
    n = len(df)
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feats)
    rep["filas_eliminadas_inf_nulos"] = int(n - len(df))
    n = len(df)
    df = df.drop_duplicates(subset=feats + ["familia"])
    rep["filas_eliminadas_duplicados"] = int(n - len(df))

    # d) submuestreo por binario
    tope = cfg["datos"].get("max_ventanas_por_clase")
    if tope:
        sel = []
        for fam, sub in df.groupby("familia"):
            if len(sub) <= tope:
                sel.append(sub)
                continue
            tam = sub.groupby("grupo").size().sample(frac=1.0, random_state=semilla)
            acum = tam.cumsum()
            grupos = acum[acum <= tope].index
            if len(grupos) == 0:
                grupos = tam.index[:1]
            sel.append(sub[sub["grupo"].isin(grupos)])
            log.info("%s: submuestreo por binario %d -> %d ventanas", fam, len(sub), int(sel[-1].shape[0]))
        df = pd.concat(sel, ignore_index=True)

    # e) partición agrupada
    es_test, detalle, agrupada = particion_agrupada(df, cfg["preprocesamiento"]["proporcion_prueba"], rng)
    if not agrupada:
        log.warning("Alguna familia tiene menos de 10 binarios identificables: revise la columna de grupo.")
    train, test = df[~es_test].reset_index(drop=True), df[es_test].reset_index(drop=True)
    assert set(train["grupo"]).isdisjoint(set(test["grupo"])), "Fuga: un binario aparece en train y test"

    destino = ruta(cfg["datos"]["dir_processed"])
    destino.mkdir(parents=True, exist_ok=True)
    train.to_parquet(destino / "train.parquet", index=False)
    test.to_parquet(destino / "test.parquet", index=False)
    guardar_json({"caracteristicas": feats, "clases": sorted(df["familia"].unique())}, destino / "caracteristicas.json")

    rep.update({
        "n_caracteristicas": len(feats),
        "caracteristicas": feats,
        "particion_agrupada_por_binario": bool(agrupada),
        "binarios_por_familia": detalle,
        "ventanas_train": train["familia"].value_counts().to_dict(),
        "ventanas_test": test["familia"].value_counts().to_dict(),
    })
    guardar_json(rep, "results/exploracion/reporte_preprocesamiento.json")
    log.info("Train: %d ventanas | Test: %d ventanas | %d características", len(train), len(test), len(feats))


if __name__ == "__main__":
    main()
