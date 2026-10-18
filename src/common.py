"""Utilidades comunes del experimento (configuración, rutas, detección de columnas y E/S).

Referencia en el paper: Sección 3.3 (Preprocesamiento) y Sección 4 (Desarrollo).
"""
from __future__ import annotations

import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[1]

# Nombres habituales de columnas de etiqueta, binario (grupo) y arquitectura
CANDIDATAS_ETIQUETA = ["family", "label", "malware_family", "class", "category", "familia", "etiqueta"]
CANDIDATAS_GRUPO = ["sha256", "hash", "sha_256", "sample", "sample_id", "binary", "md5", "file_hash", "filename", "file"]
CANDIDATAS_ARQ = ["arch", "architecture", "arquitectura", "cpu_arch"]
ARQUITECTURAS = ["mipsel", "mips", "arm", "x86"]


def cargar_config(ruta: str | Path | None = None) -> dict:
    ruta = Path(ruta) if ruta else RAIZ / "config.yaml"
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta(rel: str | Path) -> Path:
    """Convierte una ruta relativa a la raíz del repositorio en absoluta."""
    p = Path(rel)
    return p if p.is_absolute() else RAIZ / p


def logger(nombre: str) -> logging.Logger:
    log = logging.getLogger(nombre)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)


def guardar_json(obj, destino: str | Path) -> None:
    destino = ruta(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def leer_json(origen: str | Path):
    with open(ruta(origen), encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _buscar(columnas, candidatas):
    low = {c.lower().strip(): c for c in columnas}
    for cand in candidatas:
        if cand in low:
            return low[cand]
    for cand in candidatas:  # coincidencia parcial
        for k, v in low.items():
            if cand in k:
                return v
    return None


def detectar_columna_etiqueta(columnas, cfg) -> str | None:
    return cfg["datos"].get("columna_etiqueta") or _buscar(columnas, CANDIDATAS_ETIQUETA)


def detectar_columna_grupo(columnas, cfg) -> str | None:
    return cfg["datos"].get("columna_grupo") or _buscar(columnas, CANDIDATAS_GRUPO)


def detectar_columna_arquitectura(columnas, cfg) -> str | None:
    return cfg["datos"].get("columna_arquitectura") or _buscar(columnas, CANDIDATAS_ARQ)


def arquitectura_desde_nombre(nombre: str) -> str | None:
    n = nombre.lower()
    for a in ARQUITECTURAS:  # mipsel antes que mips
        if re.search(rf"(^|[^a-z]){a}([^a-z]|$)", n):
            return a.upper() if a != "mipsel" else "MIPSEL"
    return None


def normalizar_familia(valor, cfg) -> str:
    """Normaliza la familia de malware (p. ej. 'bashlite' -> 'Gafgyt')."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "Unknown"
    v = str(valor).strip()
    sinon = {k.lower(): val for k, val in (cfg["datos"].get("sinonimos") or {}).items()}
    clave = v.lower().replace("-", "_").replace(" ", "_")
    if clave in sinon:
        return sinon[clave]
    for k, val in sinon.items():
        if k in clave:
            return val
    return v


def columnas_excluidas(columnas, cfg) -> list[str]:
    pats = (cfg["preprocesamiento"].get("excluir_regex") or []) + (cfg["preprocesamiento"].get("excluir_atajos") or [])
    fuera = []
    for c in columnas:
        for p in pats:
            if re.search(p, str(c).strip(), flags=re.IGNORECASE):
                fuera.append(c)
                break
    return fuera


def listar_archivos(cfg) -> list[Path]:
    base = ruta(cfg["datos"]["dir_raw"])
    archivos = []
    for patron in cfg["datos"]["patrones"]:
        archivos += sorted(base.glob(patron))
    # Si existen a la vez x.csv y x.csv.gz (copia comprimida del repositorio), se usa solo x.csv
    nombres = {a.as_posix() for a in archivos}
    archivos = [a for a in dict.fromkeys(archivos)
                if not (a.name.endswith(".csv.gz") and a.as_posix()[:-3] in nombres)]
    filtro = (cfg["datos"].get("filtro_nombre") or "").lower()
    if filtro:
        prior = [a for a in archivos if filtro in a.name.lower()]
        if prior:
            archivos = prior
    return archivos
