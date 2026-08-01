# Predicción de cancelaciones de reservas hoteleras

Trabajo final para la materia Machine Learning de la Maestría en Ciencia de
Datos de la Universidad Católica del Uruguay.

## Descripción

El proyecto desarrolla un sistema de clasificación para estimar el riesgo de
cancelación de reservas hoteleras. El objetivo es evaluar si las probabilidades
generadas por modelos de Machine Learning permiten priorizar acciones
preventivas y mejorar la gestión de recursos e ingresos del hotel.

El trabajo no se limitará a comparar métricas predictivas. También analizará
las consecuencias de falsos positivos y falsos negativos, el costo de las
intervenciones y la selección de un threshold operativo acorde con distintos
escenarios de negocio.

## Problema y propuesta de valor

Una reserva cancelada puede provocar que una habitación quede vacía si el
hotel no logra volver a comercializarla. Sin embargo, intervenir sobre todas
las reservas también puede generar costos, descuentos innecesarios o fricción
con clientes que no tenían intención de cancelar.

El modelo producirá una probabilidad de cancelación para cada reserva. Esa
probabilidad podrá utilizarse para priorizar una capacidad limitada de acciones
preventivas, como solicitar confirmación, ofrecer una alternativa de
reprogramación o requerir determinadas garantías. La acción concreta y sus
costos se definirán durante el análisis de negocio.

## Datos

Se utilizan los datasets *Hotel Booking Demand* publicados por Nuno António,
Ana de Almeida y Luís Nunes. Los datos fueron extraídos de los sistemas de
gestión de dos hoteles reales de Portugal:

- `H1.csv`: resort hotel ubicado en la región del Algarve, con 40.060
  reservas.
- `H2.csv`: city hotel ubicado en Lisboa, con 79.330 reservas.

En conjunto contienen 119.390 reservas con fecha de llegada entre julio de
2015 y agosto de 2017. Se incluyen reservas concretadas, canceladas y
*no-shows*. Los autores eliminaron los elementos que permitían identificar a
los hoteles o a sus clientes.

Los archivos originales se conservan sin modificaciones en `data/raw` y fueron
obtenidos del material suplementario de la publicación científica.

- Artículo: [Hotel booking demand datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/)
- Datos originales: [material suplementario](https://ars.els-cdn.com/content/image/1-s2.0-S2352340918315191-mmc2.zip)
- DOI: [`10.1016/j.dib.2018.11.126`](https://doi.org/10.1016/j.dib.2018.11.126)
- Licencia: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)

## Metodología general

El proyecto seguirá las siguientes etapas:

1. Análisis exploratorio y auditoría de calidad de los datos.
2. Definición del momento de predicción y revisión de posibles fuentes de
   *data leakage*.
3. Análisis de las variables originales y formulación de hipótesis.
4. Preprocesamiento y feature engineering mediante pipelines.
5. Construcción de baselines.
6. Comparación de diferentes familias de modelos.
7. Ajuste de hiperparámetros e interpretabilidad.
8. Evaluación mediante métricas técnicas y escenarios de negocio.
9. Selección del threshold operativo.
10. Entrenamiento y documentación del modelo o los modelos finales.

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

## Notebooks

Los notebooks documentarán las distintas etapas del análisis y los
experimentos. Cada modelo podrá tener más de una iteración para registrar de
forma clara los cambios en features, preprocesamiento e hiperparámetros.

Al final del proyecto se incluirá uno o más notebooks identificados como
finales. Estos contendrán el entrenamiento reproducible del modelo o los
modelos seleccionados y concentrarán solamente las decisiones definitivas del
proyecto.

El orden de ejecución y la descripción individual de cada notebook se agregarán
cuando la estructura experimental haya quedado cerrada.

## Instalación

```bash
python -m venv .venv
pip install -r requirements.txt
```

La versión de Python, las instrucciones completas de reproducción y el punto
de entrada del entrenamiento final se documentarán al cerrar el proyecto.
