# Predicción de cancelaciones de reservas hoteleras

Trabajo final para la materia Machine Learning de la Maestría en Ciencia de
Datos de la Universidad Católica del Uruguay.

## Descripción

El proyecto desarrolla y compara modelos de clasificación para estimar el
riesgo de cancelación de reservas hoteleras. Además de evaluar su rendimiento
predictivo, analiza si las probabilidades generadas permiten priorizar acciones
preventivas con capacidad limitada.

La evaluación separa el objetivo técnico, predecir `Cancelled`, del objetivo
operativo de identificar eventos especialmente sensibles: `No-Show` y
cancelaciones registradas entre cero y siete días antes de la llegada. Esta
distinción evita interpretar automáticamente una buena métrica predictiva como
un beneficio económico.

## Datos

Se utilizan los datasets *Hotel Booking Demand* publicados por Nuno António,
Ana de Almeida y Luís Nunes. Los datos proceden de los sistemas de gestión de
dos hoteles reales de Portugal:

- `H1.csv`: resort hotel de la región del Algarve, con 40.060 reservas.
- `H2.csv`: city hotel de Lisboa, con 79.330 reservas.

En conjunto contienen 119.390 reservas con llegadas entre julio de 2015 y
agosto de 2017. Los archivos originales se conservan sin modificaciones en
`data/raw`.

- Artículo: [Hotel booking demand datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/)
- Datos originales: [material suplementario](https://ars.els-cdn.com/content/image/1-s2.0-S2352340918315191-mmc2.zip)
- DOI: [`10.1016/j.dib.2018.11.126`](https://doi.org/10.1016/j.dib.2018.11.126)
- Licencia: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)

## Metodología

El desarrollo incluye:

1. auditoría de calidad y análisis exploratorio;
2. definición del momento de predicción y control de *data leakage*;
3. análisis, reducción e ingeniería de variables;
4. baseline con `DummyClassifier` y Logistic Regression;
5. validación cruzada agrupada sobre 2015--2016;
6. holdout temporal sobre 2017;
7. comparación de Logistic Regression, Decision Tree, Random Forest,
   XGBoost, KNN y una red neuronal MLP;
8. evaluación operativa a igual capacidad;
9. análisis económico mediante escenarios conservador, intermedio y
   favorable extremo.

El holdout temporal se mantuvo fuera de las decisiones de selección de
variables e hiperparámetros.

## Principales resultados

XGBoost obtuvo el mejor ROC AUC temporal para cancelación general: `0.867291`.
Logistic Regression mejorada alcanzó ROC AUC `0.855017`, el mayor F1
(`0.713461`) y la mejor priorización económica entre los candidatos evaluados.
La MLP logró el mayor Recall con threshold 0,5 (`0.964433`), a costa de una
cantidad elevada de falsos positivos.

Los modelos ordenaron adecuadamente las cancelaciones generales, pero no
concentraron de igual forma los `No-Show` y las cancelaciones tardías. Ninguna
política produjo ahorro bajo los escenarios conservador e intermedio. Los
resultados positivos aparecieron principalmente en el escenario favorable
extremo y dependen de supuestos que el dataset no permite verificar. Por ello,
el proyecto no presenta la intervención como una fuente de ahorro demostrada.

La principal extensión propuesta es entrenar un nuevo modelo cuyo target sea
directamente `IsCriticalCancellation`.

## Estructura del repositorio

```text
.
├── data/
│   ├── raw/             # H1.csv y H2.csv originales
│   └── processed/       # Reservado para datos derivados
├── notebooks/           # Análisis narrativo y experimentos
├── src/                 # Carga, features, preprocessing y evaluación
├── models/              # Reservado para definiciones de modelos
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── report/              # Informe modular en LaTeX
├── requirements.txt
└── README.md
```

## Guía de notebooks

Los notebooks conservan resultados ejecutados y documentan las decisiones de
cada etapa:

| Notebook | Contenido |
|---|---|
| `00_main.ipynb` | Síntesis independiente de resultados técnicos, operativos y económicos. |
| `01_eda.ipynb` | Calidad de datos, distribuciones, outliers, duplicados, target y temporalidad. |
| `02_feature_analysis.ipynb` | Leakage, asociaciones, multicolinealidad y selección inicial de variables. |
| `03_baseline_logistic_regression.ipynb` | DummyClassifier, Logistic Regression base y primera formulación económica. |
| `04_feature_engineering.ipynb` | Creación de variables y ablaciones mediante validación cruzada. |
| `05_logistic_regression.ipynb` | Modelo lineal mejorado y análisis de coeficientes. |
| `06_decision_tree_feature_importance.ipynb` | Importancias, poda de variables y Decision Tree final. |
| `07_random_forest.ipynb` | Comparación de conjuntos y búsqueda de Random Forest. |
| `08_xgboost.ipynb` | Búsqueda regularizada y evaluación final de XGBoost. |
| `09_knn.ipynb` | Evaluación de KNN sobre una muestra de desarrollo. |
| `10_neural_network.ipynb` | Construcción gradual y evaluación de una MLP. |
| `11_operational_analysis.ipynb` | Reconstrucción de candidatos, comparación a igual capacidad y escenarios económicos. |

`00_main.ipynb` no entrena modelos: utiliza valores fijos validados para contar
la historia completa sin depender de la ejecución de los demás notebooks.

## Instalación

Desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Los notebooks resuelven la raíz del proyecto para importar las funciones de
`src`. Deben ejecutarse con un kernel asociado al entorno `.venv`.

## Informe

El informe se encuentra en `report/main.tex` y organiza introducción, datos,
metodología, resultados y conclusiones en archivos separados dentro de
`report/sections`.
