import pandas as pd
import numpy as np

CSV_RESULTADOS = "todo_con_resultados_17.csv"
CSV_DESCUELGUE = "todo_con_prob_descuelgue.csv"

OUT_CSV = "todo_con_resultados_prob_final_venta_precontacto_future.csv"

print("Leyendo CSVs...")

df_resultados = pd.read_csv(CSV_RESULTADOS, low_memory=False)
df_descuelgue = pd.read_csv(CSV_DESCUELGUE, low_memory=False)

# =========================
# 1) FILTRAR DESCUELGUE
# =========================

df_desc_f = df_descuelgue[df_descuelgue["camp_total_descuelgues"] > 0].copy()

print("\nFilas resultados:", len(df_resultados))
print("Filas descuelgue filtrado:", len(df_desc_f))

# =========================
# 2) MERGE POR CLAVE REAL
# =========================

df_desc_f = df_desc_f[
    ["co_cliente", "id_opp", "prob_descuelgue_modelo"]
]

df = df_resultados.merge(
    df_desc_f,
    on=["co_cliente", "id_opp"],
    how="left"
)

missing = df["prob_descuelgue_modelo"].isna().sum()

print("\nFilas sin prob_descuelgue:", missing)

# =========================
# 3) PROB FINAL
# =========================

df["prob_descuelgue_modelo"] = pd.to_numeric(df["prob_descuelgue_modelo"], errors="coerce")
df["prob_venta_modelo"] = pd.to_numeric(df["prob_venta_modelo"], errors="coerce")

df["prob_final_venta_precontacto"] = (
    df["prob_descuelgue_modelo"] * df["prob_venta_modelo"]
)

# =========================
# 4) DECILES
# =========================

df = df.sort_values(
    "prob_final_venta_precontacto",
    ascending=False
).reset_index(drop=True)

df["decil_prob_final"] = pd.qcut(
    df.index,
    10,
    labels=False
) + 1

# =========================
# 5) LIFT
# =========================

if "ganada" in df.columns:

    tasa_global = df["ganada"].mean()

    tabla = df.groupby("decil_prob_final")["ganada"].agg(
        count="count",
        tasa_venta="mean",
        ventas="sum"
    )

    tabla["lift_vs_media"] = tabla["tasa_venta"] / tasa_global

    print("\nLIFT POR DECILES")
    print(tabla)

import matplotlib.pyplot as plt
import numpy as np

# =========================
# GRÁFICO 1: TASA DE VENTA POR DECIL
# =========================

plt.figure()

plt.bar(
    tabla.index.astype(str),
    tabla["tasa_venta"]
)

plt.title("Tasa de venta por decil (1 = TOP)")
plt.xlabel("Decil")
plt.ylabel("Tasa de venta")

plt.tight_layout()
plt.show()


# =========================
# GRÁFICO 2: CURVA ACUMULADA DE VENTAS
# =========================

df_eval = df.sort_values(
    "prob_final_venta_precontacto",
    ascending=False
).reset_index(drop=True)

ventas_totales = df_eval["ganada"].sum()

df_eval["ventas_acum"] = df_eval["ganada"].cumsum()

df_eval["pct_clientes"] = np.arange(1, len(df_eval) + 1) / len(df_eval)

df_eval["pct_ventas"] = (
    df_eval["ventas_acum"] / ventas_totales
    if ventas_totales > 0 else 0
)

plt.figure()

plt.plot(
    df_eval["pct_clientes"],
    df_eval["pct_ventas"],
    label="Modelo"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    label="Aleatorio"
)

plt.title("Curva acumulada de captación de ventas")
plt.xlabel("% clientes llamados")
plt.ylabel("% ventas captadas")

plt.legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

# =========================
# AUC ROC
# =========================

y_true = df["ganada"]
y_score = df["prob_final_venta_precontacto"]

roc_auc = roc_auc_score(y_true, y_score)

print("\nAUC ROC:", roc_auc)


# =========================
# PR AUC
# =========================

precision, recall, _ = precision_recall_curve(y_true, y_score)

pr_auc = auc(recall, precision)

print("PR AUC:", pr_auc)

# =========================
# 6) GUARDAR
# =========================

df.to_csv(OUT_CSV, index=False)

print("\nCSV generado:", OUT_CSV)