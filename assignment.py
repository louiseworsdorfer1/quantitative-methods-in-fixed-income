from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CSV = DATA / "LW_monthly_1972-2024.csv"

df = pd.read_csv(CSV, sep = ';')
