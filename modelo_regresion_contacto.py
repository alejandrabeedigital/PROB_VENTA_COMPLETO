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
ARCHIVO_OUT = "todo_con_prob_descuelgue.csv"
TARGET = "target_descuelgue"

# =========================
# 1) CARGA
# =========================
df_raw = pd.read_csv(ARCHIVO_IN, low_memory=False)
print(f"Filas al cargar: {len(df_raw):,}")

# Usar target_descuelgue ya creado en nuevasvars5.csv
df_raw[TARGET] = (
    df_raw[TARGET]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({
        "TRUE": 1,
        "FALSE": 0,
        "1": 1,
        "0": 0
    })
)

df_raw = df_raw[df_raw[TARGET].isin([0, 1])].copy()
df_raw[TARGET] = df_raw[TARGET].astype(int)

print(f"Filas tras definir target 0/1: {len(df_raw):,}")

# =========================
# 2) UNIVERSO MODELABLE
# =========================
df = df_raw.copy()

print(f"Filas tras definir universo modelable: {len(df):,}")

# =========================
# 3) FEATURES
# =========================

features_num = [
]

features_cat = [
    "sin_gmb",
    "movil",
    "ct_merclie",
    "con_web",
    "con_local",
    "cat_contact",
    "outcome_sin_con_pred",
    "ant_empresa",
    "algun_contacto"
]

# --- convertir algun_contacto a 0/1 si viene como TRUE/FALSE
if "algun_contacto" in df.columns:
    df["algun_contacto"] = (
        df["algun_contacto"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({
            "TRUE": 1,
            "FALSE": 0,
            "1": 1,
            "0": 0
        })
    )

# --- asegurar tipos numéricos
for col in features_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# evitar problemas sklearn con pandas StringDtype
for c in features_cat:
    if c in df.columns:
        df[c] = df[c].astype(object)

# =========================
# 3.1 DESCRIPTIVOS
# =========================

print("\n==============================")
print("  DESCRIPTIVOS (TOTAL y %)")
print("==============================\n")

def guardar_descriptivo_categorica(df_in, col, out_name):
    vc = df_in[col].astype(object).where(df_in[col].notna(), "NaN").value_counts(dropna=False)

    out = vc.rename_axis("valor").reset_index(name="total")
    out["porcentaje"] = out["total"] / out["total"].sum() * 100

    out.to_csv(out_name, index=False)

    print(f"✅ Guardado: {out_name} (n={out['total'].sum():,})")

for c in features_cat:
    if c in df.columns:
        guardar_descriptivo_categorica(df, c, f"descriptivos_{c}.csv")


def guardar_descriptivo_numerica_deciles(df_in, col, out_name):
    s = pd.to_numeric(df_in[col], errors="coerce")

    tmp = pd.DataFrame({col: s}).dropna()

    if tmp.empty:
        print(f"⚠️ {col}: sin datos.")
        return

    bins = pd.qcut(tmp[col], 10, duplicates="drop")

    vc = bins.value_counts().sort_index()

    out = vc.rename_axis("bin").reset_index(name="total")
    out["porcentaje"] = out["total"] / out["total"].sum() * 100

    out.to_csv(out_name, index=False)

    print(f"✅ Guardado: {out_name}")

for c in features_num:
    if c in df.columns:
        guardar_descriptivo_numerica_deciles(df, c, f"descriptivos_{c}_deciles.csv")

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

transformers = []

if len(features_num) > 0:
    transformers.append(
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), features_num)
    )

if len(features_cat) > 0:
    transformers.append(
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), features_cat)
    )

preprocess = ColumnTransformer(
    transformers=transformers
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
df_eval[TARGET] = y_test.values
df_eval["score"] = proba_test

df_eval = df_eval.sort_values("score", ascending=False).reset_index(drop=True)

df_eval["decil"] = pd.qcut(df_eval.index, 10, labels=False, duplicates="drop") + 1

tabla = df_eval.groupby("decil")[TARGET].agg(["count", "mean", "sum"])
tabla = tabla.rename(columns={"mean": "tasa_descuelgue", "sum": "descuelgues"})

tabla["lift_vs_media"] = tabla["tasa_descuelgue"] / tasa_global

print("\n--- LIFT POR DECILES ---")
print(tabla)

# =========================
# 8) PROBABILIDAD PARA TODOS
# =========================

X_all = df[features_num + features_cat]

df["prob_descuelgue_modelo"] = pipeline.predict_proba(X_all)[:, 1]

# =========================
# 9) GRÁFICOS
# =========================

plt.figure()
plt.bar(tabla.index.astype(str), tabla["tasa_descuelgue"])
plt.title("Tasa de descuelgue por decil (1 = TOP)")
plt.xlabel("Decil")
plt.ylabel("Tasa de descuelgue")
plt.tight_layout()
plt.show()

totales_objetivo = df_eval[TARGET].sum()
df_eval["objetivo_acum"] = df_eval[TARGET].cumsum()
df_eval["pct_clientes"] = np.arange(1, len(df_eval) + 1) / len(df_eval)
df_eval["pct_objetivo"] = df_eval["objetivo_acum"] / totales_objetivo if totales_objetivo > 0 else 0

plt.figure()
plt.plot(df_eval["pct_clientes"], df_eval["pct_objetivo"])
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Curva acumulada de captación de descuelgues")
plt.xlabel("% clientes llamados")
plt.ylabel("% descuelgues captados")
plt.tight_layout()
plt.show()

# =========================
# 10) COEFICIENTES
# =========================

feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
coefs = pipeline.named_steps["clf"].coef_[0]

df_coefs = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs,
    "abs_coef": np.abs(coefs)
}).sort_values("abs_coef", ascending=False)

df_coefs.to_csv("coeficientes_modelo_descuelgue.csv", index=False)

print("\n--- TOP COEFICIENTES ---")
print(df_coefs.head(20))

print("\n✅ Guardado: coeficientes_modelo_descuelgue.csv")

# =========================
# 11) GUARDAR
# =========================

cols_out = [c for c in df.columns if c != "prob_descuelgue_modelo"] + ["prob_descuelgue_modelo"]

df[cols_out].to_csv(ARCHIVO_OUT, index=False)

print(f"\n✅ Guardado: {ARCHIVO_OUT}")
print(f"Filas guardadas: {len(df):,}")