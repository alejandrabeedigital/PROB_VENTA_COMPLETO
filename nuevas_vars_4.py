import pandas as pd

archivo_entrada = "nuevasvars3.csv"
archivo_salida = "nuevasvars4.csv"

# Leer CSV
df = pd.read_csv(
    archivo_entrada,
    low_memory=False
)

# Limpiar nombres de columnas
df.columns = df.columns.str.strip()

# Verificar columna
if "ct_sociedad" not in df.columns:
    raise ValueError(f"No existe la columna 'ct_sociedad' en {archivo_entrada}")

# -------------------------
# Normalizar ct_sociedad
# -------------------------
ct_norm = (
    df["ct_sociedad"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0+$", "", regex=True)  # elimina .0 .00 etc
    .str.replace(r"\.$", "", regex=True)    # elimina punto final
)

# -------------------------
# Crear aut_o_no
# -------------------------
def clasificar_sociedad(x):
    if x in ["00", "0"]:
        return "AUTONOMO"
    elif x == "22":
        return "DESCONOCIDO"
    else:
        return "NO_AUTONOMO"

df["aut_o_no"] = ct_norm.apply(clasificar_sociedad)

# Guardar
df.to_csv(archivo_salida, index=False, encoding="utf-8")

print(f"✅ Archivo guardado: {archivo_salida}")
print(f"Filas totales: {len(df):,}")
print("Distribución aut_o_no:")
print(df["aut_o_no"].value_counts())