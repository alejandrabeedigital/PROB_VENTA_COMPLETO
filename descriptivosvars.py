import pandas as pd
import numpy as np

# =========================
# Cargar datos
# =========================
archivo = "nuevasvars5.csv"

df = pd.read_csv(
    archivo,
    sep=",",
    encoding="utf-8",
    low_memory=False
)

# =========================
# Preprocesado cat_contact
# =========================
if "cat_contact" in df.columns:
    df["cat_contact"] = df["cat_contact"].fillna(0)
    df["cat_contact"] = df["cat_contact"].replace(0, "NOCONTACT")
    df["cat_contact"] = df["cat_contact"].astype("category")

# =========================
# Crear variable algun_contacto
# TRUE si camp_total_descuelgues > 0
# FALSE si camp_total_descuelgues == 0
# =========================
if "total_descuelgues" in df.columns:
    df["camp_total_descuelgues"] = df["camp_total_descuelgues"].fillna(0)
    df["total_descuelgues"] = pd.to_numeric(df["total_descuelgues"], errors="coerce")

    df["algun_contacto"] = np.where(
        df["total_descuelgues"] > 0,
        "TRUE",
        "FALSE"
    )

    df["algun_contacto"] = df["algun_contacto"].astype("category")

# =========================
# NUEVO TARGET
# TRUE si camp_total_descuelgues > 0
# FALSE si camp_total_descuelgues == 0
# =========================
if "camp_total_descuelgues" in df.columns:
    df["camp_total_descuelgues"] = df["camp_total_descuelgues"].fillna(0)

    df["target_descuelgue"] = np.where(
        df["camp_total_descuelgues"] > 0,
        "TRUE",
        "FALSE"
    )

    df["target_descuelgue"] = df["target_descuelgue"].astype("category")

# =========================
# Guardar dataset modificado
# =========================
df.to_csv("nuevasvars5.csv", index=False, encoding="utf-8-sig")

# =========================
# Variables para descriptivos
# =========================
variables = [
    "sin_gmb",
    "movil",
    "ct_merclie",
    "con_web",
    "con_local",
    "algun_contacto",
    "cat_contact",
    "outcome_sin_con_pred",
    "ant_empresa"
    "algun_contacto",
    "target_descuelgue"
]

# =========================
# Descriptivos: conteo y %
# incluyendo NaN
# =========================
resultados = []

for var in variables:
    if var not in df.columns:
        print(f"[AVISO] La variable '{var}' no existe en el dataset")
        continue

    conteos = df[var].value_counts(dropna=False)
    porcentajes = df[var].value_counts(dropna=False, normalize=True) * 100

    tabla = pd.DataFrame({
        "variable": var,
        "valor": conteos.index,
        "n": conteos.values,
        "porcentaje": porcentajes.values
    })

    tabla["valor"] = tabla["valor"].astype(object)
    tabla["valor"] = tabla["valor"].where(~pd.isna(tabla["valor"]), "NaN")
    tabla["porcentaje"] = tabla["porcentaje"].round(2)

    resultados.append(tabla)

descriptivos = pd.concat(resultados, ignore_index=True)

# =========================
# Guardar descriptivos
# =========================
descriptivos.to_csv("descriptivos_nuevasvars5.csv", index=False, encoding="utf-8-sig")

print("Hecho.")
print("Archivo actualizado: nuevasvars5.csv")
print("Archivo descriptivos: descriptivos_nuevasvars5.csv")
print(descriptivos)