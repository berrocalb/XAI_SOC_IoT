"""Paso 6. Figuras y tablas del artículo.

Referencia en el paper: Sección 5 (Resultados): Tabla 3, Tabla 4 y Figuras 2 a 5.

Salidas:
  results/figuras/fig2_matriz_confusion.png     (Sección 5.1)
  results/figuras/fig3_metricas_explicacion.png (Sección 5.2: Δp@k, NC80, CT3 por condición)
  results/figuras/fig4_inversion_estabilidad.png(Sección 5.2: TID@k, EE y RP)
  results/figuras/fig5_ejemplos_alertas.png     (Sección 5.3: ejemplo de explicación por familia)
  results/tablas/tabla_resultados_explicacion.csv (Tabla 4)
  results/tablas/tabla_indicadores_vi.csv        (costo y concordancia)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import leer_json, logger, ruta

log = logger("06_figuras")

# Colores por condición: C0 neutro; C1 y C2 = primeras ranuras categóricas validadas (azul, naranja)
COL = {"C0": "#8a8984", "C1": "#2a78d6", "C2": "#eb6834"}
NOM = {"C0": "C0 · MDI global", "C1": "C1 · TreeSHAP", "C2": "C2 · LIME"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def estilo(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_confusion():
    cm = pd.read_csv(ruta("results/tablas/matriz_confusion.csv"), index_col=0)
    norm = cm.div(cm.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(3.4, 2.9), dpi=300)
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=0)
    ax.set_yticks(range(len(cm.index)), cm.index)
    ax.set_xlabel("Familia predicha"); ax.set_ylabel("Familia real")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = norm.values[i, j]
            ax.text(j, i, f"{v:.2f}\n({cm.values[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 else INK)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta("results/figuras/fig2_matriz_confusion.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_metricas(reg):
    mets = [("dp_k", "Δp@k (mayor = más fiel)"), ("nc80", "NC80 (menor = más concisa)"), ("ct3", "CT3 % (mayor = más concisa)")]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), dpi=300)
    for ax, (m, tit) in zip(axs, mets):
        datos = [reg.loc[reg.condicion == c, m].dropna().to_numpy() for c in ("C0", "C1", "C2")]
        bp = ax.boxplot(datos, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": INK2, "lw": 0.8},
                        capprops={"color": INK2, "lw": 0.8})
        for patch, c in zip(bp["boxes"], ("C0", "C1", "C2")):
            patch.set_facecolor(COL[c]); patch.set_alpha(0.85); patch.set_edgecolor("white"); patch.set_linewidth(1.5)
        ax.set_xticks([1, 2, 3], ["C0\nMDI", "C1\nTreeSHAP", "C2\nLIME"])
        ax.set_title(tit, fontsize=9, color=INK)
        estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta("results/figuras/fig3_metricas_explicacion.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_inversion_estabilidad(reg, umbral=0.80):
    g = reg.groupby("condicion").agg(tid=("inversion", "mean"), ee=("ee", "median"), rp=("rp", "median"))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), dpi=300)
    conds = ["C0", "C1", "C2"]
    vals = [100 * g.loc[c, "tid"] for c in conds]
    bars = axs[0].bar(conds, vals, color=[COL[c] for c in conds], width=0.55, edgecolor="white", linewidth=2)
    for b, v in zip(bars, vals):
        axs[0].text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f} %", ha="center", fontsize=8, color=INK)
    axs[0].set_xticks(range(3), [NOM[c] for c in conds]); axs[0].set_ylabel("TID@k (%)"); axs[0].set_ylim(0, 105)
    axs[0].set_title("Inversión de la decisión", fontsize=9); estilo(axs[0])
    x = np.arange(2); w = 0.36
    for j, (c, desp) in enumerate((("C1", -w / 2), ("C2", w / 2))):
        v = [g.loc[c, "ee"], g.loc[c, "rp"]]
        bb = axs[1].bar(x + desp, v, width=w, color=COL[c], edgecolor="white", linewidth=2, label=NOM[c])
        for b, val in zip(bb, v):
            axs[1].text(b.get_x() + b.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=8, color=INK)
    axs[1].axhline(umbral, color=INK2, lw=0.8, ls="--")
    from matplotlib.transforms import blended_transform_factory
    axs[1].text(1.01, umbral, "umbral\n0,80", fontsize=7, color=INK2, ha="left", va="center",
                transform=blended_transform_factory(axs[1].transAxes, axs[1].transData))
    axs[1].set_xticks(x, ["EE (entre ejecuciones)", "RP (perturbación)"]); axs[1].set_ylim(0, 1.32)
    axs[1].set_ylabel("Jaccard top-k (mediana)"); axs[1].set_title("Estabilidad", fontsize=9)
    axs[1].legend(frameon=False, fontsize=7, loc="upper left", ncol=2); estilo(axs[1])
    fig.tight_layout()
    fig.savefig(ruta("results/figuras/fig4_inversion_estabilidad.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_ejemplos():
    topk = leer_json("results/registro/explicaciones_topk.json")
    vistos, ejemplos = set(), []
    for a in topk:
        if a["familia_real"] == a["familia_pred"] and a["familia_real"] not in vistos:
            vistos.add(a["familia_real"]); ejemplos.append(a)
    if not ejemplos:
        return
    fig, axs = plt.subplots(len(ejemplos), 2, figsize=(7.0, 1.7 * len(ejemplos)), dpi=300, squeeze=False)
    for i, a in enumerate(ejemplos):
        for j, c in enumerate(("C1", "C2")):
            ax = axs[i, j]
            items = a[c][::-1]
            v = [it["atribucion"] for it in items]
            ax.barh([it["caracteristica"][:28] for it in items], v, color=COL[c], height=0.6)
            ax.axvline(0, color=INK2, lw=0.6)
            ax.set_title(f"{a['alerta_id']} · {a['familia_real']} · {NOM[c]}", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ruta("results/figuras/fig5_ejemplos_alertas.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def tablas(reg, vi):
    t = reg.groupby("condicion").agg(dp_k_mediana=("dp_k", "median"), tid_pct=("inversion", lambda s: 100 * s.mean()),
                                     nc80_mediana=("nc80", "median"), ct3_mediana=("ct3", "median"),
                                     ee_mediana=("ee", "median"), rp_mediana=("rp", "median")).round(3)
    t.to_csv(ruta("results/tablas/tabla_resultados_explicacion.csv"))
    mem = leer_json("results/registro/memoria.json")["memoria"]
    v = pd.DataFrame({
        "indicador": ["TME TreeSHAP (ms, mediana)", "TME LIME (ms, mediana)", "MP TreeSHAP (MB, solo memoria Python; el núcleo C++ no se mide)", "MP LIME (MB)",
                      "CJ SHAP-LIME (mediana)", "Spearman SHAP-LIME (mediana)"],
        "valor": [vi["tme_shap_ms"].median(), vi["tme_lime_ms"].median(), mem["TreeSHAP"]["memoria_pico_mb"],
                  mem["LIME"]["memoria_pico_mb"], vi["cj"].median(), vi["spearman"].median()]}).round(3)
    v.to_csv(ruta("results/tablas/tabla_indicadores_vi.csv"), index=False)
    print(t.to_string()); print(v.to_string(index=False))


def main():
    ruta("results/figuras").mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(ruta("results/registro/registro_observacion.csv"))
    vi = pd.read_csv(ruta("results/registro/registro_vi.csv"))
    fig_confusion(); fig_metricas(reg); fig_inversion_estabilidad(reg); fig_ejemplos(); tablas(reg, vi)
    log.info("Figuras en results/figuras y tablas en results/tablas")


if __name__ == "__main__":
    main()
