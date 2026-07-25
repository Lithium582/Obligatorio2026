# Obligatorio 2026 — Machine Learning

Trabajo final para la materia Machine Learning de la Maestría en Ciencia de
Datos de la Universidad Católica del Uruguay.

## Estado

El repositorio contiene la plantilla inicial del proyecto. El problema, el
dataset, la variable objetivo y la métrica principal todavía deben definirse a
partir de los lineamientos del trabajo.

La infraestructura general se reutiliza de un proyecto anterior, pero no se
trasladan features, hiperparámetros ni decisiones específicas de aquel
dataset.

## Estructura

```text
.
├── data/
│   ├── raw/            # Datos originales, sin modificar
│   └── processed/      # Datos derivados, si fueran necesarios
├── notebooks/          # Análisis narrativo y experimentos
├── src/                # Funciones reutilizables
├── models/             # Definiciones de modelos, cuando se seleccionen
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── report/             # Informe en LaTeX
└── lineamientos_trabajo_final.pdf
```

## Secuencia de trabajo propuesta

1. Obtener y documentar el origen de los datos.
2. Definir el problema, la variable objetivo y la métrica principal.
3. Auditar la calidad y estructura del dataset.
4. Realizar el análisis exploratorio.
5. Analizar las features originales y formular hipótesis.
6. Implementar y evaluar feature engineering sin fuga de información.
7. Construir baselines y comparar modelos mediante una estrategia común.
8. Ajustar los modelos seleccionados y realizar la evaluación final.
9. Integrar resultados, limitaciones y conclusiones en el informe.

## Instalación

```bash
python -m venv .venv
pip install -r requirements.txt
```

Los valores específicos del proyecto deben completarse en `src/config.py`
cuando se haya seleccionado el dataset.
