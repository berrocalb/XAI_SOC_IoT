"""Paso 3. Entrenamiento y evaluación del detector Random Forest; importancia MDI (condición C0).

Referencia en el paper: Sección 4.1 (Detector: Random Forest), ecuaciones (8) y (9);
Sección 5.1 (Desempeño del detector, Tabla 3 y Figura 2).

Salidas:
  results/modelo/random_forest.joblib
  results/tablas/desempeno_detector.csv      (precisión, exhaustividad, F1 y TFP por familia; macro)
  results/tablas/matriz_confusion.csv
  results/tablas/importancia_mdi.csv          (vector de la condición C0)
  results/modelo/metricas_detector.json
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from common import cargar_config, fijar_semilla, guardar_json, leer_json, logger, ruta

log = logger("03_detector")


def main():
    cfg = cargar_config()
    fijar_semilla(cfg["semilla"])
    dproc = ruta(cfg["datos"]["dir_processed"])
    meta = leer_json(dproc / "caracteristicas.json")
    feats, clases = meta["caracteristicas"], meta["clases"]
    train = pd.read_parquet(dproc / "train.parquet")
    test = pd.read_parquet(dproc / "test.parquet")

    m = cfg["modelo"]
    rf = RandomForestClassifier(n_estimators=m["n_estimators"], max_depth=m["max_depth"],
                                min_samples_leaf=m["min_samples_leaf"], class_weight=m["class_weight"],
                                n_jobs=m["n_jobs"], random_state=cfg["semilla"])
    t0 = time.perf_counter()
    rf.fit(train[feats].to_numpy(np.float32), train["familia"].to_numpy())
    t_fit = time.perf_counter() - t0
    log.info("Entrenamiento: %.1f s", t_fit)

    Xte, yte = test[feats].to_numpy(np.float32), test["familia"].to_numpy()
    t0 = time.perf_counter()
    pred = rf.predict(Xte)
    t_pred = (time.perf_counter() - t0) / len(Xte)

    etiquetas = list(rf.classes_)
    p, r, f1, sop = precision_recall_fscore_support(yte, pred, labels=etiquetas, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=etiquetas)
    total = cm.sum()
    filas = []
    for i, c in enumerate(etiquetas):
        fp = cm[:, i].sum() - cm[i, i]
        vn = total - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        filas.append({"familia": c, "precision": p[i], "exhaustividad": r[i], "f1": f1[i],
                      "tfp": fp / (fp + vn) if (fp + vn) else 0.0, "soporte": int(sop[i])})
    tabla = pd.DataFrame(filas)
    macro = {"familia": "Macro", "precision": tabla["precision"].mean(), "exhaustividad": tabla["exhaustividad"].mean(),
             "f1": f1_score(yte, pred, average="macro"), "tfp": tabla["tfp"].mean(), "soporte": int(tabla["soporte"].sum())}
    tabla = pd.concat([tabla, pd.DataFrame([macro])], ignore_index=True)

    ruta("results/tablas").mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta("results/tablas/desempeno_detector.csv"), index=False)
    pd.DataFrame(cm, index=etiquetas, columns=etiquetas).to_csv(ruta("results/tablas/matriz_confusion.csv"))
    mdi = pd.DataFrame({"caracteristica": feats, "mdi": rf.feature_importances_}).sort_values("mdi", ascending=False)
    mdi.to_csv(ruta("results/tablas/importancia_mdi.csv"), index=False)
    joblib.dump(rf, ruta("results/modelo/random_forest.joblib"), compress=3)
    guardar_json({"exactitud": accuracy_score(yte, pred), "f1_macro": macro["f1"], "tfp_macro": macro["tfp"],
                  "tiempo_entrenamiento_s": t_fit, "tiempo_prediccion_ms_por_ventana": 1000 * t_pred,
                  "clases": etiquetas, "n_train": len(train), "n_test": len(test),
                  "hiperparametros": m}, "results/modelo/metricas_detector.json")
    log.info("F1 macro = %.4f | TFP macro = %.4f", macro["f1"], macro["tfp"])
    print(tabla.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
