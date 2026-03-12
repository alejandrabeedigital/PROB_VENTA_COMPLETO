import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

ARCHIVO = "todo_con_prob_descuelgue.csv"
TARGET = "target_descuelgue"

# =========================
# 1) CARGA
# =========================
df = pd.read_csv(ARCHIVO, low_memory=False)
print(f"Filas leídas: {len(df):,}")

# Asegurar target 0/1 desde TRUE/FALSE o 0/1
df[TARGET] = (
    df[TARGET]
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

df = df[df[TARGET].isin([0, 1])].copy()
df[TARGET] = df[TARGET].astype(int)

# =========================
# 2) VARIABLES
# =========================
# ant_empresa en tu dataset es categórica (ej: "0_2_años")
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

features = features_num + features_cat

# =========================
# 2.1) TIPOS / LIMPIEZA MÍNIMA
# =========================

# numéricas
for c in features_num:
    if c in df.columns:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )

# categóricas -> object
for c in features_cat:
    if c in df.columns:
        df[c] = df[c].astype(object)

# asegurar que existen todas
faltan = [c for c in features + [TARGET] if c not in df.columns]
if faltan:
    raise ValueError(f"Faltan columnas en el CSV: {faltan}")

# dataset final
df_model = df[features + [TARGET]].dropna().copy()
print(f"Filas usadas para inferencia: {len(df_model):,}")
print(f"Tasa base: {df_model[TARGET].mean():.6f}")

# =========================
# 3) DUMMIES
# =========================
X = pd.get_dummies(df_model[features], drop_first=True)
y = df_model[TARGET]

# todo numérico
X = X.astype(float)

# Escalar solo numéricas si existieran
for col in features_num:
    if col in X.columns:
        mu = X[col].mean()
        sd = X[col].std()
        if sd is not None and sd > 0:
            X[col] = (X[col] - mu) / sd

# quitar columnas constantes o sin variación
cols_sin_var = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
if cols_sin_var:
    print("\n⚠️ Columnas eliminadas por no tener variación:")
    print(cols_sin_var)
    X = X.drop(columns=cols_sin_var)

# constante
X = sm.add_constant(X, has_constant="add")

print("\nTipos de X:")
print(X.dtypes)

print(f"\nShape X: {X.shape}")
print(f"Shape y: {y.shape}")

# =========================
# 4) REGRESIÓN LOGÍSTICA
# =========================
model = sm.Logit(y, X)

# Optimizador más estable
result = model.fit(method="lbfgs", maxiter=5000, disp=True)

print("\n==============================")
print("      RESUMEN COMPLETO")
print("==============================\n")
print(result.summary())

# =========================
# 5) ODDS RATIOS
# =========================
odds_ratios = pd.DataFrame({
    "variable": result.params.index,
    "coef": result.params.values,
    "odds_ratio": np.exp(result.params.values),
    "p_value": result.pvalues.values
}).sort_values("p_value")

print("\n==============================")
print("      ODDS RATIOS")
print("==============================\n")
print(odds_ratios)

odds_ratios.to_csv("odds_ratios_modelo_contacto.csv", index=False)
print("\n✅ Guardado: odds_ratios_modelo_contacto.csv")

# =========================
# 6) EFECTOS MARGINALES
# =========================
marginal = result.get_margeff()

print("\n==============================")
print("    EFECTOS MARGINALES")
print("==============================\n")
print(marginal.summary())

mfx_table = marginal.summary_frame().reset_index()
mfx_table.to_csv("efectos_marginales_modelo_contacto.csv", index=False)
print("✅ Guardado: efectos_marginales_modelo_contacto.csv")

# =========================
# 7) INTERPRETACIÓN AUTOMÁTICA
# =========================
print("\n==============================")
print("    INTERPRETACIÓN CLAVE")
print("==============================\n")

for var, coef, p in zip(result.params.index, result.params.values, result.pvalues.values):
    if var == "const":
        continue
    efecto = "AUMENTA" if coef > 0 else "DISMINUYE"
    signif = "SIGNIFICATIVO" if p < 0.05 else "NO SIGNIFICATIVO"
    print(f"{var}: {efecto} probabilidad de contacto | p-value={p:.4f} | {signif}")

# =========================
# 8) GRÁFICO EFECTOS MARGINALES
# =========================
mfx = marginal.margeff
variables = marginal.summary_frame().index

df_mfx = pd.DataFrame({
    "variable": variables,
    "dy_dx": mfx
})

df_mfx = df_mfx[df_mfx["variable"] != "const"].copy()

# multiplicar por 10000 para lectura más cómoda
df_mfx["dy_dx_10000"] = df_mfx["dy_dx"] * 10000

df_mfx = df_mfx.sort_values("dy_dx_10000")

plt.figure(figsize=(10, 8))
plt.barh(df_mfx["variable"], df_mfx["dy_dx_10000"])
plt.xlabel("Cambio en probabilidad de contacto (por 10,000 clientes)")
plt.title("Efectos marginales del modelo logístico de contacto")
plt.tight_layout()
plt.show()

# =========================
# 9) PREDICCIÓN EN MUESTRA
# =========================
df_model["prob_contacto_logit_inferencia"] = result.predict(X)

df_model[features + [TARGET, "prob_contacto_logit_inferencia"]].to_csv(
    "predicciones_inferencia_contacto.csv",
    index=False
)

print("✅ Guardado: predicciones_inferencia_contacto.csv")

# =========================
# 10) TABLA FINAL PARA EXCEL
# =========================
# =========================
# 7) TABLA FINAL PARA EXCEL
# =========================
mfx_summary = marginal.summary_frame().reset_index()

mfx_summary = mfx_summary.rename(columns={
    "index": "variable",
    "dy/dx": "dy_dx",
    "Pr(>|z|)": "p_value"
})

cols_necesarias = ["variable", "dy_dx", "p_value"]
faltan_cols = [c for c in cols_necesarias if c not in mfx_summary.columns]
if faltan_cols:
    raise ValueError(f"No encuentro estas columnas en marginal.summary_frame(): {faltan_cols}")

df_excel = mfx_summary[cols_necesarias].copy()

df_excel["dy_dx_10000"] = df_excel["dy_dx"] * 10000
df_excel["significancia"] = np.where(df_excel["p_value"] < 0.05, "S", "NS")

df_excel = df_excel[["variable", "dy_dx", "dy_dx_10000", "p_value", "significancia"]]
df_excel = df_excel.sort_values(["p_value", "variable"]).reset_index(drop=True)

# -------- versión numérica para guardar bien --------
df_excel.to_csv(
    "tabla_efectos_marginales_excel_contacto.csv",
    index=False,
    sep=";",
    decimal=","
)

# -------- versión texto bonita para copiar/pegar en Excel --------
df_excel_copy = df_excel.copy()

df_excel_copy["dy_dx"] = df_excel_copy["dy_dx"].map(lambda x: f"{x:.6f}".replace(".", ","))
df_excel_copy["dy_dx_10000"] = df_excel_copy["dy_dx_10000"].map(lambda x: f"{x:.3f}".replace(".", ","))
df_excel_copy["p_value"] = df_excel_copy["p_value"].map(lambda x: f"{x:.6e}".replace(".", ","))

print("\n==============================")
print(" TABLA FINAL PARA COPIAR A EXCEL")
print("==============================\n")

print(df_excel_copy.to_csv(index=False, sep=";"))

print("✅ Guardado: tabla_efectos_marginales_excel_contacto.csv")