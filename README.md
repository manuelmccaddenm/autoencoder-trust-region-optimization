# Autoencoder lineal por Región de Confianza

Optimización de un autoencoder lineal `R^d -> R^k -> R^d` mediante
Región de Confianza con subproblema Dogleg y aproximación del Hessiano
inverso vía iBFGS / L-BFGS, sin recurrir a diferenciación automática.

Proyecto de Análisis Aplicado I, ITAM (MAT 24430), Primavera 2026.
Autor: Manuel McCadden Madrazo.

## Documento

[`Reporte.pdf`](Reporte.pdf) describe la formulación matemática, los
algoritmos y los resultados numéricos.

## Reproducción

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

Lo anterior corre, en orden:

1. **`main.py`** -- benchmark iBFGS vs L-BFGS sobre `d in {3, 5, 10, 100}`,
   `k=2`, `N=200` (Tabla 1 del reporte). Termina en segundos.
2. **`make_figs.py`** -- regenera las figuras del reporte
   (`figs/*.pdf`). Termina en segundos.
3. **`stress_test.py`** -- 5000 configuraciones aleatorias verificando
   ocho invariantes algoritmicos (Conclusiones del reporte). Tarda
   ~6 min.

Para omitir el stress test:

```bash
python reproduce.py fast
```

## Estructura

```
main.py             # punto de entrada del benchmark
make_figs.py        # genera las figuras
stress_test.py      # corre 5000 configuraciones aleatorias
reproduce.py        # orquesta los tres scripts anteriores
lib_data.py         # generación sintética de datos
lib_autoencoder.py  # pérdida MSE, gradientes analíticos, empaquetado
lib_dogleg.py       # subproblema Dogleg con interpolación cuadrática
lib_ibfgs.py        # actualización iBFGS sobre H_k
lib_lbfgs.py        # recursión de dos ciclos (memoria limitada)
lib_trust_region.py # bucle externo
lib_benchmark.py    # utilidad para correr el benchmark
Reporte.pdf         # reporte final
```

Las semillas están fijas en todos los experimentos; la ejecución es
determinista.
