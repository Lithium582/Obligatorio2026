from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "models"

REPORT_DIR = ROOT_DIR / "report"
REPORT_FIGURES_DIR = REPORT_DIR / "figures"
REPORT_TABLES_DIR = REPORT_DIR / "tables"

RANDOM_STATE = 42

# Variable objetivo del problema de cancelación hotelera.
TARGET_COL = "IsCanceled"
ID_COL = None
