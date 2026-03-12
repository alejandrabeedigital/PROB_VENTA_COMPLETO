import pandas as pd
import numpy as np
from datetime import datetime

ARCHIVO_IN = "nuevasvars1.csv"
ARCHIVO_OUT = "nuevasvars2.csv"

# =========================
# 1) CARGA
# =========================
df = pd.read_csv(ARCHIVO_IN, low_memory=False)
print(f"Filas cargadas: {len(df):,}")

# =========================
# 2) sin_intentos_recientes
# =========================
# TRUE si no hay intentos registrados en los últimos 6 meses
df["sin_intentos_recientes"] = df["intentos_ult6m"].isna()

import pandas as pd
import numpy as np

# =========================
# 3) ANTIGÜEDAD EMPRESA (FIX SIMPLE)
# =========================

# convertir a número
year = pd.to_numeric(df["fe_creacion_empresa"], errors="coerce")

# aceptar solo años razonables
year = year.where((year >= 1800) & (year <= pd.Timestamp.today().year))

# convertir a fecha
df["fe_creacion_empresa_dt"] = pd.to_datetime(
    year.astype("Int64").astype(str) + "-01-01",
    errors="coerce"
)

# antigüedad en años
today = pd.Timestamp.today()

df["antig_empresa_years"] = (
    (today - df["fe_creacion_empresa_dt"]).dt.days / 365.25
)

# categorías
df["ant_empresa"] = pd.cut(
    df["antig_empresa_years"],
    bins=[0, 2, 5, 10, np.inf],
    labels=["0_2_años", "2_5_años", "5_10_años", "10+_años"],
    include_lowest=True
)

# checks
print("\nDistribución ant_empresa:")
print(df["ant_empresa"].value_counts(dropna=False))

print("\nAños usados:")
print(year.value_counts(dropna=False).head(20))

# =========================
# 5) GUARDAR
# =========================
df.to_csv(ARCHIVO_OUT, index=False, encoding="utf-8")

print(f"\nArchivo guardado: {ARCHIVO_OUT}")
print(f"Filas guardadas: {len(df):,}")