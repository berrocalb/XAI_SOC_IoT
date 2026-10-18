"""Paso 4. Generación y evaluación de explicaciones por alerta (Algoritmo 1 del paper).

Referencia en el paper: Sección 3.4 (Procedimiento), Sección 3.5 (Métricas, ecuaciones 1-6),
Secciones 4.2-4.4 (TreeSHAP, LIME e integración) y Sección 5.2 (Resultados de las explicaciones).

Condiciones:
  C0 = importancia global MDI del Random Forest (pre-test, igual para todas las alertas)
  C1 = TreeSHAP (post-test)
  C2 = LIME (post-test)

Salidas (carpeta results/registro/):
  registro_observacion.csv   una fila por alerta y condición (Δp@k, inversión, NC80, CT3, EE, RP, tiempo)
  registro_vi.csv            concordancia SHAP-LIME y tiempos por alerta (indicadores de la VI)
  explicaciones_topk.json    las k características principales de cada condición por alerta
  memoria.json               memoria pico de cada explicador

Uso:
  python src/04_explicar_alertas.py            # configuración completa (n = 390, R = 10)
  python src/04_explicar_alertas.py --rapido   # prueba rápida (n pequeño, R = 3)
"""
import argparse
import time
import tracemalloc
import warnings

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from tqdm import tqdm

import shap
from common import cargar_config, fijar_semilla, guardar_json, leer_json, logger, ruta

warnings.filterwarnings("ignore")
log = logger("04_explicaciones")


# ----------------------------------------------------------------------------- métricas (Sección 3.5)
def top_k(phi, k):
    return set(np.argsort(-np.abs(phi))[:k].tolist())


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def nc80(phi):
    a = np.sort(np.abs(phi))[::-1]
    tot = a.sum()
    if tot <= 0:
        return np.nan
    return int(np.searchsorted(np.cumsum(a) / tot, 0.80 - 1e-12) + 1)


def ct3(phi):
    a = np.sort(np.abs(phi))[::-1]
    tot = a.sum()
    return np.nan if tot <= 0 else 100.0 * a[:3].sum() / tot


def fidelidad(modelo, x, phi, idx_pred, ref, k):
    """Δp@k (ec. 1) e indicador de inversión (ec. 2): se neutralizan las top-k con la mediana de entrenamiento."""
    xt = x.copy()
    sel = list(top_k(phi, k))
    xt[sel] = ref[sel]
    p0 = modelo.predict_proba(x.reshape(1, -1))[0]
    p1 = modelo.predict_proba(xt.reshape(1, -1))[0]
    return float(p0[idx_pred] - p1[idx_pred]), int(np.argmax(p1) != idx_pred)


# ----------------------------------------------------------------------------- explicadores
def shap_vector(explainer, x, idx):
    sv = explainer.shap_values(x.reshape(1, -1), check_additivity=False)
    if isinstance(sv, list):          # versiones antiguas de shap: lista por clase
        return np.asarray(sv[idx])[0]
    sv = np.asarray(sv)
    return sv[0, :, idx] if sv.ndim == 3 else sv[0]


def lime_vector(explainer, modelo, x, idx, d, n_samples, semilla):
    explainer.random_state = np.random.RandomState(semilla)
    explainer.base.random_state = explainer.random_state
    exp = explainer.explain_instance(x, modelo.predict_proba, labels=(idx,), num_features=d, num_samples=n_samples)
    phi = np.zeros(d)
    for j, w in exp.as_map()[idx]:
        phi[j] = w
    return phi


def perturbar(modelo, x, idx, sigma, lo, hi, eps, rng, intentos):
    """x' = x + ε σ ⊙ u, u ~ N(0, I), recortado al dominio de entrenamiento y con la misma clase (ec. 4)."""
    for _ in range(intentos):
        xp = np.clip(x + eps * sigma * rng.standard_normal(len(x)), lo, hi).astype(x.dtype)
        if np.argmax(modelo.predict_proba(xp.reshape(1, -1))[0]) == idx:
            return xp
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="prueba rápida con pocas alertas")
    args = ap.parse_args()
    cfg = cargar_config()
    semilla = cfg["semilla"]
    fijar_semilla(semilla)
    rng = np.random.default_rng(semilla)
    e = dict(cfg["explicacion"])
    if args.rapido:
        e.update({"alertas_por_clase": 8, "repeticiones_lime": 3, "lime_num_samples": 500, "lime_fondo": 1000})
    k, R = e["k"], e["repeticiones_lime"]

    dproc = ruta(cfg["datos"]["dir_processed"])
    feats = leer_json(dproc / "caracteristicas.json")["caracteristicas"]
    d = len(feats)
    train = pd.read_parquet(dproc / "train.parquet")
    test = pd.read_parquet(dproc / "test.parquet")
    rf = joblib.load(ruta("results/modelo/random_forest.joblib"))
    rf.set_params(n_jobs=1)  # predicciones por alerta: un solo hilo es más rápido y evita avisos de joblib
    clases = list(rf.classes_)

    Xtr = train[feats].to_numpy(np.float32)
    ref = np.median(Xtr, axis=0)
    sigma = Xtr.std(axis=0)
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)

    # Muestra estratificada de alertas por familia real (Sección 3.4)
    partes = []
    for fam, sub in test.groupby("familia"):
        n = min(e["alertas_por_clase"], len(sub))
        partes.append(sub.sample(n=n, random_state=semilla))
    alertas = pd.concat(partes).reset_index(drop=True)
    Xa = alertas[feats].to_numpy(np.float32)
    proba = rf.predict_proba(Xa)
    idx_pred = proba.argmax(axis=1)
    log.info("Alertas muestreadas: %d (%s)", len(alertas), alertas["familia"].value_counts().to_dict())

    # Explicadores
    phi_mdi = rf.feature_importances_.astype(float)
    t_shap = shap.TreeExplainer(rf)  # modalidad tree_path_dependent: sin datos de referencia, determinista
    fondo = Xtr[rng.choice(len(Xtr), size=min(e["lime_fondo"], len(Xtr)), replace=False)]
    t_lime = LimeTabularExplainer(fondo, feature_names=feats, class_names=clases, mode="classification",
                                  discretize_continuous=True, random_state=semilla)

    filas, filas_vi, topk = [], [], []
    for i in tqdm(range(len(alertas)), desc="Alertas"):
        x, c = Xa[i], int(idx_pred[i])
        base = {"alerta_id": f"ALR-{i + 1:04d}", "familia_real": alertas.loc[i, "familia"],
                "familia_pred": clases[c], "p_pred": float(proba[i, c]), "grupo": alertas.loc[i, "grupo"]}

        t0 = time.perf_counter(); phi_s = shap_vector(t_shap, x, c); t_s = 1000 * (time.perf_counter() - t0)
        phis_l, tiempos_l = [], []
        for r in range(R):
            t0 = time.perf_counter()
            phis_l.append(lime_vector(t_lime, rf, x, c, d, e["lime_num_samples"], semilla + r))
            tiempos_l.append(1000 * (time.perf_counter() - t0))
        phi_l = phis_l[0]

        # Robustez ante perturbación
        xp = perturbar(rf, x, c, sigma, lo, hi, e["epsilon"], rng, e["intentos_perturbacion"])
        if xp is not None:
            rp = {"C0": 1.0,
                  "C1": jaccard(top_k(phi_s, k), top_k(shap_vector(t_shap, xp, c), k)),
                  "C2": jaccard(top_k(phi_l, k), top_k(lime_vector(t_lime, rf, xp, c, d, e["lime_num_samples"], semilla), k))}
        else:
            rp = {"C0": np.nan, "C1": np.nan, "C2": np.nan}

        # Estabilidad entre ejecuciones (ec. 3)
        pares = [jaccard(top_k(phis_l[a], k), top_k(phis_l[b], k)) for a in range(R) for b in range(a + 1, R)]
        ee = {"C0": 1.0, "C1": 1.0, "C2": float(np.mean(pares)) if pares else np.nan}

        for cond, phi, t in (("C0", phi_mdi, 0.0), ("C1", phi_s, t_s), ("C2", phi_l, float(np.mean(tiempos_l)))):
            dp, inv = fidelidad(rf, x, phi, c, ref, k)
            filas.append({**base, "condicion": cond, "dp_k": dp, "inversion": inv, "nc80": nc80(phi),
                          "ct3": ct3(phi), "ee": ee[cond], "rp": rp[cond], "tiempo_ms": t})
        rho = spearmanr(np.abs(phi_s), np.abs(phi_l)).correlation
        filas_vi.append({**base, "cj": jaccard(top_k(phi_s, k), top_k(phi_l, k)),
                         "spearman": float(rho) if rho == rho else np.nan,
                         "tme_shap_ms": t_s, "tme_lime_ms": float(np.mean(tiempos_l))})
        topk.append({**base, **{cond: [{"caracteristica": feats[j], "atribucion": float(phi[j])}
                                       for j in np.argsort(-np.abs(phi))[:k]]
                                for cond, phi in (("C0", phi_mdi), ("C1", phi_s), ("C2", phi_l))}})

    # Memoria pico (indicador MP) sobre una submuestra de alertas
    m = {}
    sub = Xa[: min(20, len(Xa))]
    for nombre, fn in (("TreeSHAP", lambda x, c: shap_vector(t_shap, x, c)),
                       ("LIME", lambda x, c: lime_vector(t_lime, rf, x, c, d, e["lime_num_samples"], semilla))):
        tracemalloc.start()
        for j, x in enumerate(sub):
            fn(x, int(idx_pred[j]))
        _, pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        m[nombre] = {"memoria_pico_mb": pico / 2 ** 20, "alertas_medidas": len(sub)}

    dreg = ruta("results/registro")
    dreg.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(dreg / "registro_observacion.csv", index=False)
    pd.DataFrame(filas_vi).to_csv(dreg / "registro_vi.csv", index=False)
    guardar_json(topk, dreg / "explicaciones_topk.json")
    guardar_json({"memoria": m, "parametros": e, "n_alertas": len(alertas)}, dreg / "memoria.json")
    log.info("Registro guardado en %s", dreg)
    print(pd.DataFrame(filas).groupby("condicion")[["dp_k", "inversion", "nc80", "ct3", "ee", "rp"]].mean().round(3))


if __name__ == "__main__":
    main()
