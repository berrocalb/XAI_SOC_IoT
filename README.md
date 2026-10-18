# Marco de explicabilidad basado en SHAP y LIME para alertas de detección de intrusiones en SOC que monitorean redes IoT (2026)

Repositorio del experimento del artículo:

> **Marco de explicabilidad basado en SHAP y LIME para mejorar la interpretabilidad de las alertas de detección de intrusiones en SOC que monitorean redes IoT a nivel internacional, 2026**
> Autor: Jorge Luis Berrocal Barrientos. Inteligencia Artificial II (CBS06), Facultad de Ingeniería Eléctrica y Electrónica, Universidad Nacional de Ingeniería (Perú), 2026-2.

El experimento entrena un detector **Random Forest** que clasifica ventanas de tráfico de red de malware IoT por familia
(Mirai, DarkNexus y Gafgyt) y compara, sobre las mismas 390 alertas, tres condiciones de explicación:
**C0** importancia global MDI (línea base, pre-test), **C1** TreeSHAP y **C2** LIME (post-test). Mide la fidelidad,
la estabilidad, la parsimonia y el costo de las explicaciones, y contrasta las hipótesis con pruebas no paramétricas.

## Resultados principales

| Métrica (mediana, n = 390) | C0: MDI | C1: TreeSHAP | C2: LIME |
|---|--:|--:|--:|
| Fidelidad: Δp@k | 0,024 | 0,031 | 0,033 |
| Fidelidad: TID@k (%) | 41,0 | 41,8 | 41,3 |
| Estabilidad entre ejecuciones: EE | 1,000 | 1,000 | 0,504 |
| Robustez ante perturbación: RP | 1,000 | 0,667 | 0,429 |
| Parsimonia: NC80 (menor es mejor) | 9 | **7** | 12 |
| Parsimonia: CT3 (%) | 63,8 | **66,7** | 48,5 |

- Detector: exactitud 0,886; F1 macro 0,553 (Mirai 0,955; DarkNexus 0,401; Gafgyt 0,303).
- **H.E.1** (fidelidad): no se acepta. **H.E.2** (estabilidad ≥ 0,80): no se acepta.
  **H.E.3** (parsimonia): se acepta parcialmente (solo TreeSHAP; NC80 de 9 a 7, r = −0,70, p_Holm < 0,001).
- Costo: TreeSHAP 244 ms por alerta; LIME 119 ms por ejecución (× 10). Concordancia SHAP-LIME: Jaccard 0,25; Spearman 0,42.

Detalle: `results/tablas/` y `results/tablas/decision_hipotesis.md`. Registro de la corrida: `logs/ejecucion_final.log`.

## Estructura del repositorio

```
xai_soc_iot/
├── config.yaml                 parámetros del experimento (semilla 42, k = 5, R = 10, n = 390)
├── ejecutar_todo.py            ejecuta los pasos 01-06
├── src/                        scripts 00-06 y funciones comunes
├── data/raw/pcap.csv.gz        datos crudos (partición ARM de Kaggle, comprimidos, 23 MB)
├── data/processed/             train/test en Parquet y lista de características
├── results/                    exploración, métricas del detector, registro por alerta, tablas y figuras
├── docs/                       artículo, sistematización y matriz de trazabilidad
├── fuentes/                    PDF de las referencias (descarga con fuentes/descargar_fuentes.py)
├── logs/                       registro de la ejecución final
└── tests/                      generador de datos sintéticos para la prueba de humo
```

## Dataset

**CIC-YNU-IoTMal 2026** (Canadian Institute for Cybersecurity, University of New Brunswick, Canadá; Yokohama National
University, Japón). Malware IoT capturado a nivel mundial por los honeypots IoTPOT y ejecutado en sandbox (QEMU/OpenWrt).
Este estudio usa la modalidad de **tráfico de red** (ventanas con 39 características del esquema CICIoT2023) de la
partición **ARM** redistribuida en Kaggle: 737 651 ventanas y 5 599 binarios.

- Kaggle: https://www.kaggle.com/datasets/raneemababneh/iot-malware-network-traffic-dataset-cic-ynu-2026
- Página oficial: https://www.unb.ca/cic/datasets/ynu-iot-2026.html
- Artículo: Dadkhah, S., et al. (2026). *Information Systems, 140*, 102722. https://doi.org/10.1016/j.is.2026.102722

Se incluyen Mirai, DarkNexus y Gafgyt. Se excluyen las ejecuciones benignas (generadas con apoyo de modelos de lenguaje),
las etiquetas Unknown y Generic, y las familias con menos de cinco binarios (Tsunami y Agent).

El CSV original (194 MB) supera el límite de GitHub, por eso se versiona comprimido (`data/raw/pcap.csv.gz`); los scripts
lo leen directamente. Si descarga el CSV desde Kaggle y lo deja en `data/raw/`, se usa el CSV y se ignora el `.gz`.

## 1. Preparar el entorno de Python

Requisitos: **Python 3.10 a 3.13** (probado con 3.12 y 3.13; `lime` todavía no instala en 3.14 o superior), 8 GB de RAM.

**Windows (PowerShell)**
```powershell
cd xai_soc_iot
py install 3.12                      # solo si no tiene Python 3.12
py -V:3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Linux / macOS**
```bash
cd xai_soc_iot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Conda (alternativa)**: `conda env create -f environment.yml` y luego `conda activate xai-soc-iot`.
Versiones exactas: `pip install -r requirements-lock.txt`.

## 2. Datos

Los datos ya vienen en `data/raw/pcap.csv.gz`. Para descargarlos de nuevo desde Kaggle, guarde su token
(`kaggle.json`) en `~/.kaggle/` o `C:\Users\<usuario>\.kaggle\` y ejecute `python src/00_descargar_datos.py`.

## 3. Ejecutar el experimento

```bash
python ejecutar_todo.py            # pasos 01 a 06 (unos 13 min en una CPU de 2 núcleos)
python ejecutar_todo.py --rapido   # prueba rápida del paso 04 (pocas alertas)
python ejecutar_todo.py --desde 4  # retoma desde el paso 4
```

| Paso | Script | Salidas principales | Sección del artículo |
|---|---|---|---|
| 00 | `src/00_descargar_datos.py` | `data/raw/` | 3.2 |
| 01 | `src/01_explorar_dataset.py` | `results/exploracion/resumen_exploracion.{json,md}` | 3.2 (Tabla 2) |
| 02 | `src/02_preprocesar.py` | `data/processed/*.parquet`, `results/exploracion/reporte_preprocesamiento.json` | 3.3 |
| 03 | `src/03_entrenar_detector.py` | `results/modelo/metricas_detector.json`, `results/tablas/desempeno_detector.csv`, `matriz_confusion.csv`, `importancia_mdi.csv` | 4.1; 5.1 (Tabla 3, Figura 2) |
| 04 | `src/04_explicar_alertas.py` | `results/registro/registro_observacion.csv`, `registro_vi.csv`, `explicaciones_topk.json`, `memoria.json` | 3.4-3.5; 4.2-4.4; 5.2-5.4 |
| 05 | `src/05_analisis_estadistico.py` | `results/tablas/contrastes.csv`, `estadistica_descriptiva.csv`, `decision_hipotesis.md` | 3.6; 5.2 (Tabla 4); 7 |
| 06 | `src/06_figuras_tablas.py` | `results/figuras/fig2..fig5.png`, `results/tablas/tabla_*.csv` | 5 |

En el artículo, la Figura 4 corresponde al archivo `fig5_ejemplos_alertas.png`; `fig4_inversion_estabilidad.png` es
una figura complementaria para la exposición. `results/registro/registro_observacion.csv` es el **registro de
observación estructurada** de la sistematización (una fila por alerta y condición).

## 4. Documentos y trazabilidad

- `docs/Paper_borrador_v2.pdf` y `.docx`: artículo (APA 7, IMRyD). Los comentarios del `.docx` indican, por sección,
  las fuentes y los archivos del repositorio que la respaldan.
- `docs/Sistematizacion_SHAP_LIME_SOC_IoT_Internacional_v4.pdf` y `.docx`: sistematización y matriz de consistencia.
- `docs/trazabilidad.md` y `docs/Matriz_trazabilidad_paper.xlsx`: correspondencia sección → referencia → archivo.
- `fuentes/`: `python fuentes/descargar_fuentes.py` descarga los PDF de acceso abierto según `manifest_fuentes.csv`;
  los que el sitio no permite descargar se listan en `fuentes/pendientes_descarga_manual.csv`.

## 5. Limitaciones conocidas

- Solo la partición ARM y la modalidad de red; Gafgyt tiene 5 binarios (3 de entrenamiento y 2 de prueba).
- El detector es débil para las familias minoritarias (gran parte se confunde con Mirai).
- La fidelidad por neutralización con la mediana de entrenamiento se satura (dominada por Mirai).
- `tracemalloc` no mide la memoria del núcleo C++ de TreeSHAP.

## 6. Prueba de humo (solo para verificar el código)

`tests/generar_datos_prueba.py` crea datos **sintéticos** con la estructura del dataset; no se usan en el artículo.

## Licencia

Código bajo licencia MIT (`LICENSE`). Los datos pertenecen a sus autores (ver la página oficial del CIC) y se redistribuyen solo con fines académicos.
