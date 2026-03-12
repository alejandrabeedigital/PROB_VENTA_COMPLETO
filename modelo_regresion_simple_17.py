import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ARCHIVO_IN = "nuevasvars5.csv"
ARCHIVO_OUT = "todo_con_resultados_17.csv"
ARCHIVO_DESC = "descvarspostmodelo.csv"
TARGET = "ganada"

# =========================
# 1) CARGA
# =========================
df_raw = pd.read_csv(ARCHIVO_IN, low_memory=False)
print(f"Filas al cargar: {len(df_raw):,}")

df_raw[TARGET] = pd.to_numeric(df_raw[TARGET], errors="coerce")
df_raw = df_raw[df_raw[TARGET].isin([0, 1])].copy()
df_raw[TARGET] = df_raw[TARGET].astype(int)

print(f"Filas tras filtrar target 0/1: {len(df_raw):,}")

# =========================
# 2) FILTRO DESCUELGUES CAMPAÑA
# =========================
df_raw["camp_total_descuelgues"] = pd.to_numeric(df_raw["camp_total_descuelgues"], errors="coerce")
df = df_raw[df_raw["camp_total_descuelgues"].fillna(0) > 0].copy()

print(f"Filas tras filtrar camp_total_descuelgues > 0: {len(df):,}")

# =========================
# 3) FEATURES
# =========================

features_num = [
    "q_rk_score"
]

features_cat = [
    "ct_merclie",
    "excliente_cat",
    "origen_sc_o_no",
    "con_web",
    "sin_gmb",
    "gmb_sin_owner",
    "movil",

    # NUEVAS VARIABLES
    "sin_intentos_recientes",
    "ant_empresa",
    "outcome_sin_con_pred"
]

# --- asegurar tipos numéricos
for col in features_num:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# evitar problemas sklearn con pandas StringDtype
for c in features_cat:
    if c in df.columns:
        df[c] = df[c].astype(object)

# =========================
# 3.1 DESCRIPTIVOS (UN SOLO CSV)
# =========================

print("\n==============================")
print("  DESCRIPTIVOS (TOTAL y %)")
print("==============================\n")

desc_list = []

for c in features_cat:
    if c in df.columns:

        vc = df[c].astype(object).where(df[c].notna(), "NaN").value_counts(dropna=False)

        out = vc.rename_axis("categoria").reset_index(name="total")
        out["pct"] = out["total"] / out["total"].sum() * 100
        out["variable"] = c

        desc_list.append(out[["variable", "categoria", "total", "pct"]])


for c in features_num:
    if c in df.columns:

        s = pd.to_numeric(df[c], errors="coerce")

        tmp = pd.DataFrame({c: s}).dropna()

        if not tmp.empty:

            bins = pd.qcut(tmp[c], 10, duplicates="drop")

            vc = bins.value_counts().sort_index()

            out = vc.rename_axis("categoria").reset_index(name="total")
            out["pct"] = out["total"] / out["total"].sum() * 100
            out["variable"] = c

            desc_list.append(out[["variable", "categoria", "total", "pct"]])

desc_final = pd.concat(desc_list, ignore_index=True)

desc_final.to_csv(ARCHIVO_DESC, index=False)

print(f"✅ Descriptivos guardados en {ARCHIVO_DESC}")

# =========================
# 4) DATASET MODELADO
# =========================

cols_modelo = features_num + features_cat + [TARGET]

faltan = [c for c in cols_modelo if c not in df.columns]
if faltan:
    raise ValueError(f"Faltan columnas en el CSV: {faltan}")

df_model = df[cols_modelo].copy()

pos = int(df_model[TARGET].sum())
neg = int((df_model[TARGET] == 0).sum())

tasa_global = pos / (pos + neg)

print(f"Filas para modelar: {len(df_model):,}")
print(f"Positivos={pos}, Negativos={neg}, tasa={tasa_global:.6f}")

# =========================
# 5) PIPELINE
# =========================

preprocess = ColumnTransformer(

    transformers=[

        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), features_num),

        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), features_cat)
    ]
)

clf = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    solver="lbfgs"
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", clf)
])

X = df_model[features_num + features_cat]
y = df_model[TARGET]

use_stratify = y.nunique() == 2 and y.value_counts().min() >= 2

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y if use_stratify else None
)

pipeline.fit(X_train, y_train)

# =========================
# 6) MÉTRICAS
# =========================

proba_test = pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, proba_test)
auprc = average_precision_score(y_test, proba_test)

print(f"\nAUC: {auc:.4f}")
print(f"AUPRC: {auprc:.6f}")

# =========================
# 7) RANKING Y LIFT
# =========================

df_eval = X_test.copy()
df_eval["ganada"] = y_test.values
df_eval["score"] = proba_test

df_eval = df_eval.sort_values("score", ascending=False).reset_index(drop=True)

df_eval["decil"] = pd.qcut(df_eval.index, 10, labels=False) + 1

tabla = df_eval.groupby("decil")["ganada"].agg(["count", "mean", "sum"])
tabla = tabla.rename(columns={"mean": "tasa_venta", "sum": "ventas"})

tabla["lift_vs_media"] = tabla["tasa_venta"] / tasa_global

print("\n--- LIFT POR DECILES ---")
print(tabla)

# =========================
# 8) PROBABILIDAD PARA TODOS
# =========================

X_all = df[features_num + features_cat]

df["prob_venta_modelo"] = pipeline.predict_proba(X_all)[:, 1]

# =========================
# 9) GRÁFICOS
# =========================
plt.figure()
plt.bar(tabla.index.astype(str), tabla["tasa_venta"])
plt.title("Tasa de venta por decil (1 = TOP)")
plt.xlabel("Decil")
plt.ylabel("Tasa de venta")
plt.tight_layout()
plt.show()

ventas_totales = df_eval["ganada"].sum()
df_eval["ventas_acum"] = df_eval["ganada"].cumsum()
df_eval["pct_clientes"] = np.arange(1, len(df_eval) + 1) / len(df_eval)
df_eval["pct_ventas"] = df_eval["ventas_acum"] / ventas_totales if ventas_totales > 0 else 0

plt.figure()
plt.plot(df_eval["pct_clientes"], df_eval["pct_ventas"])
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Curva acumulada de captación de ventas")
plt.xlabel("% clientes llamados")
plt.ylabel("% ventas captadas")
plt.tight_layout()
plt.show()

# =========================
# 10) GUARDAR
# =========================

cols_out = [c for c in df.columns if c != "prob_venta_modelo"] + ["prob_venta_modelo"]

df[cols_out].to_csv(ARCHIVO_OUT, index=False)

print(f"\n✅ Guardado: {ARCHIVO_OUT}")
print(f"Filas guardadas: {len(df):,}")