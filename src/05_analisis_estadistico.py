"""Paso 5. Contraste de hipótesis (Sección 3.6 del paper, ecuación 7).

Referencia en el paper: Sección 3.6 (Análisis estadístico), Sección 5.2 (Tabla 4) y Sección 7 (Conclusiones,
validación de H.E.1, H.E.2 y H.E.3).

  H.E.1 (fidelidad):  Friedman sobre Δp@k; Q de Cochran sobre la inversión (TID@k); post hoc Wilcoxon / McNemar + Holm.
  H.E.2 (estabilidad): Wilcoxon de una muestra (unilateral) de EE y RP de C1 y C2 frente a 0,80.
  H.E.3 (parsimonia): Friedman sobre NC80 y CT3; post hoc Wilcoxon + Holm.
Salidas: results/tablas/estadistica_descriptiva.csv, results/tablas/contrastes.csv,
         results/tablas/decision_hipotesis.md
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import cargar_config, logger, ruta

log = logger("05_estadistica")
PARES = [("C1", "C0"), ("C2", "C0"), ("C1", "C2")]


def holm(pvals):
    """Corrección de Holm (1979): devuelve p ajustados."""
    p = np.asarray(pvals, float)
    orden = np.argsort(p)
    m = len(p)
    aj = np.empty(m)
    acum = 0.0
    for rango, i in enumerate(orden):
        acum = max(acum, (m - rango) * p[i])
        aj[i] = min(1.0, acum)
    return aj


def wilcoxon_r(a, b=None, alternativa="two-sided", mu=0.0):
    """Wilcoxon de rangos con signo y tamaño del efecto r = Z/sqrt(n) (Fritz et al., 2012).

    Z se obtiene con la aproximación normal del estadístico W+ (con corrección por empates), de modo que su signo
    indica la dirección de la diferencia (a - b > 0 si Z > 0)."""
    x = np.asarray(a, float) - (np.asarray(b, float) if b is not None else mu)
    x = x[~np.isnan(x)]
    x = x[x != 0]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, n
    res = stats.wilcoxon(x, alternative=alternativa)
    rangos = stats.rankdata(np.abs(x))
    w_mas = rangos[x > 0].sum()
    media = n * (n + 1) / 4
    _, cuentas = np.unique(rangos, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (cuentas ** 3 - cuentas).sum() / 48
    z = (w_mas - media) / np.sqrt(var) if var > 0 else 0.0
    return float(res.pvalue), float(z / np.sqrt(n)), n


def cochran_q(matriz):
    """Prueba Q de Cochran para k condiciones dicotómicas pareadas."""
    X = np.asarray(matriz, float)
    k = X.shape[1]
    col, fila, N = X.sum(axis=0), X.sum(axis=1), X.sum()
    den = k * N - (fila ** 2).sum()
    if den == 0:
        return 0.0, 1.0
    q = (k - 1) * (k * (col ** 2).sum() - N ** 2) / den
    return float(q), float(stats.chi2.sf(q, k - 1))


def mcnemar(a, b):
    """McNemar exacto (binomial) para dos condiciones dicotómicas pareadas."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(n10, n01 + n10, 0.5).pvalue)


def main():
    cfg = cargar_config()
    alfa = cfg["estadistica"]["alfa"]
    umbral = cfg["explicacion"]["umbral_estabilidad"]
    reg = pd.read_csv(ruta("results/registro/registro_observacion.csv"))
    ancho = {m: reg.pivot(index="alerta_id", columns="condicion", values=m) for m in
             ["dp_k", "inversion", "nc80", "ct3", "ee", "rp", "tiempo_ms"]}

    desc = (reg.groupby("condicion")[["dp_k", "inversion", "nc80", "ct3", "ee", "rp", "tiempo_ms"]]
            .agg(["median", "mean", "std"]).round(4))
    desc.to_csv(ruta("results/tablas/estadistica_descriptiva.csv"))

    filas = []
    # Friedman + Wilcoxon/Holm para Δp@k, NC80 y CT3
    for met, hip, alt in (("dp_k", "H.E.1", "greater"), ("nc80", "H.E.3", "less"), ("ct3", "H.E.3", "greater")):
        w = ancho[met][["C0", "C1", "C2"]].dropna()
        chi, p = stats.friedmanchisquare(w["C0"], w["C1"], w["C2"])
        W = chi / (len(w) * (3 - 1))
        filas.append({"hipotesis": hip, "metrica": met, "prueba": "Friedman", "comparacion": "C0-C1-C2",
                      "estadistico": chi, "p": p, "p_holm": np.nan, "efecto": W, "efecto_tipo": "W de Kendall", "n": len(w)})
        res = []
        for a, b in PARES:
            altp = alt if b == "C0" else "two-sided"
            pv, r, n = wilcoxon_r(w[a], w[b], altp)
            res.append((a, b, pv, r, n, altp))
        aj = holm([x[2] for x in res])
        for (a, b, pv, r, n, altp), pa in zip(res, aj):
            filas.append({"hipotesis": hip, "metrica": met, "prueba": f"Wilcoxon ({altp})", "comparacion": f"{a} vs {b}",
                          "estadistico": np.nan, "p": pv, "p_holm": pa, "efecto": r, "efecto_tipo": "r", "n": n})

    # Cochran Q + McNemar/Holm para la inversión de la decisión (TID@k)
    w = ancho["inversion"][["C0", "C1", "C2"]].dropna()
    q, p = cochran_q(w.to_numpy())
    filas.append({"hipotesis": "H.E.1", "metrica": "inversion", "prueba": "Q de Cochran", "comparacion": "C0-C1-C2",
                  "estadistico": q, "p": p, "p_holm": np.nan, "efecto": np.nan, "efecto_tipo": "", "n": len(w)})
    pv = [mcnemar(w[a], w[b]) for a, b in PARES]
    for (a, b), p1, pa in zip(PARES, pv, holm(pv)):
        filas.append({"hipotesis": "H.E.1", "metrica": "inversion", "prueba": "McNemar exacto", "comparacion": f"{a} vs {b}",
                      "estadistico": np.nan, "p": p1, "p_holm": pa, "efecto": np.nan, "efecto_tipo": "", "n": len(w)})

    # H.E.2: estabilidad frente al umbral 0,80
    for met in ("ee", "rp"):
        for cond in ("C1", "C2"):
            x = ancho[met][cond].dropna()
            if x.nunique() <= 1:
                pv_, r_, n_ = (np.nan, np.nan, len(x))
            else:
                pv_, r_, n_ = wilcoxon_r(x, mu=umbral, alternativa="greater")
            filas.append({"hipotesis": "H.E.2", "metrica": met, "prueba": "Wilcoxon una muestra (> 0,80)",
                          "comparacion": cond, "estadistico": float(np.median(x)) if len(x) else np.nan,
                          "p": pv_, "p_holm": np.nan, "efecto": r_, "efecto_tipo": "r", "n": n_})

    con = pd.DataFrame(filas)
    con.to_csv(ruta("results/tablas/contrastes.csv"), index=False)

    # Decisión sobre las hipótesis
    def sig(h, met, comp):
        f = con[(con.hipotesis == h) & (con.metrica == met) & (con.comparacion == comp)]
        return (not f.empty) and (f["p_holm"].fillna(f["p"]).iloc[0] < alfa)
    med = {(m, c): float(ancho[m][c].median()) for m in ("ee", "rp") for c in ("C1", "C2")}
    nombre = {"C1": "TreeSHAP (C1)", "C2": "LIME (C2)"}
    md = ["# Decisión sobre las hipótesis específicas", "", f"Nivel de significación α = {alfa}. "
          "Cada hipótesis se evalúa por método; se acepta para el marco si se cumple en ambos métodos.", ""]
    for h, texto, cond in (
        ("H.E.1", "fidelidad (Δp@k mayor que C0, Wilcoxon unilateral + Holm)", lambda c: sig("H.E.1", "dp_k", f"{c} vs C0")),
        ("H.E.2", f"estabilidad (medianas de EE y RP ≥ {umbral})", lambda c: med[("ee", c)] >= umbral and med[("rp", c)] >= umbral),
        ("H.E.3", "parsimonia (NC80 menor y CT3 mayor que C0, Wilcoxon unilateral + Holm)",
         lambda c: sig("H.E.3", "nc80", f"{c} vs C0") and sig("H.E.3", "ct3", f"{c} vs C0"))):
        res = {c: cond(c) for c in ("C1", "C2")}
        global_ = "se acepta" if all(res.values()) else ("se acepta parcialmente" if any(res.values()) else "no se acepta")
        md.append(f"- **{h}** – {texto}: {global_} "
                  f"({'; '.join(f'{nombre[c]}: ' + ('sí' if v else 'no') for c, v in res.items())}).")
    md += ["", "Medianas de estabilidad: " + ", ".join(f"{m.upper()}-{c} = {v:.3f}" for (m, c), v in med.items()) + "."]
    ruta("results/tablas/decision_hipotesis.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Contrastes guardados en results/tablas/contrastes.csv")
    print("\n".join(md))


if __name__ == "__main__":
    main()
