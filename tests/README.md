# Prueba de humo

`generar_datos_prueba.py` crea datos SINTÉTICOS con la estructura del tráfico de CIC-YNU-IoTMal 2026 para verificar
que los scripts se ejecutan de principio a fin. No se usan en el artículo.

```bash
python tests/generar_datos_prueba.py --destino data/raw/prueba
python ejecutar_todo.py --rapido
```

Después de la prueba, borre `data/raw/prueba/`, `data/processed/*.parquet` y los archivos generados en `results/`.
