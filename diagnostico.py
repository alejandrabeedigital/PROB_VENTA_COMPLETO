import pandas as pd

# =========================
# CONFIG
# =========================
CSV_RESULTADOS = "todo_con_resultados_17.csv"
CSV_DESCUELGUE = "todo_con_prob_descuelgue.csv"

# Posibles claves candidatas para hacer merge/comparación
POSSIBLE_KEYS = [
    "id",
    "id_cliente",
    "customer_id",
    "lead_id",
    "telefono",
    "phone",
    "telefono_1",
    "telefono_formateado",
    "nif",
    "documento",
    "cod_cliente",
    "geo_code"
]

# =========================
# 1) CARGA
# =========================
print("Leyendo CSVs...")
df_resultados = pd.read_csv(CSV_RESULTADOS, low_memory=False)
df_descuelgue = pd.read_csv(CSV_DESCUELGUE, low_memory=False)

print("\n=========================")
print("RESUMEN INICIAL")
print("=========================")
print(f"Filas resultados : {len(df_resultados):,}")
print(f"Filas descuelgue : {len(df_descuelgue):,}")
print(f"Cols resultados  : {len(df_resultados.columns)}")
print(f"Cols descuelgue  : {len(df_descuelgue.columns)}")

print("\nPrimeras columnas en resultados:")
print(df_resultados.columns.tolist()[:30])

print("\nPrimeras columnas en descuelgue:")
print(df_descuelgue.columns.tolist()[:30])

# =========================
# 2) COLUMNAS COMUNES
# =========================
common_cols = sorted(set(df_resultados.columns).intersection(df_descuelgue.columns))
print("\n=========================")
print("COLUMNAS COMUNES")
print("=========================")
print(f"Número de columnas comunes: {len(common_cols)}")
print(common_cols[:100])

# =========================
# 3) BUSCAR CLAVE CANDIDATA
# =========================
candidate_keys = [c for c in POSSIBLE_KEYS if c in common_cols]

print("\n=========================")
print("CLAVES CANDIDATAS EN COMÚN")
print("=========================")
if candidate_keys:
    print(candidate_keys)
else:
    print("No se encontró ninguna clave candidata de la lista.")
    print("Busca manualmente una columna identificadora común.")

# =========================
# 4) ANALIZAR CADA CLAVE CANDIDATA
# =========================
def analizar_clave(df1, df2, key, nombre1="resultados", nombre2="descuelgue"):
    print("\n" + "=" * 60)
    print(f"ANÁLISIS DE CLAVE: {key}")
    print("=" * 60)

    s1 = df1[key].astype(str).str.strip()
    s2 = df2[key].astype(str).str.strip()

    null_1 = s1.isna().sum()
    null_2 = s2.isna().sum()

    print(f"Nulos en {nombre1}: {null_1:,}")
    print(f"Nulos en {nombre2}: {null_2:,}")

    dup_1 = s1.duplicated().sum()
    dup_2 = s2.duplicated().sum()

    print(f"Duplicados en {nombre1}: {dup_1:,}")
    print(f"Duplicados en {nombre2}: {dup_2:,}")

    set1 = set(s1.dropna())
    set2 = set(s2.dropna())

    solo_1 = set1 - set2
    solo_2 = set2 - set1
    inter = set1 & set2

    print(f"Valores únicos en {nombre1}: {len(set1):,}")
    print(f"Valores únicos en {nombre2}: {len(set2):,}")
    print(f"Intersección              : {len(inter):,}")
    print(f"Solo en {nombre1}         : {len(solo_1):,}")
    print(f"Solo en {nombre2}         : {len(solo_2):,}")

    if len(solo_1) > 0:
        print(f"Ejemplos solo en {nombre1}: {list(sorted(solo_1))[:10]}")
    if len(solo_2) > 0:
        print(f"Ejemplos solo en {nombre2}: {list(sorted(solo_2))[:10]}")

for key in candidate_keys:
    analizar_clave(df_resultados, df_descuelgue, key)

# =========================
# 5) HIPÓTESIS: FILTRO camp_total_descuelgues > 0
# =========================
print("\n=========================")
print("HIPÓTESIS camp_total_descuelgues > 0")
print("=========================")

for nombre, df in [("resultados", df_resultados), ("descuelgue", df_descuelgue)]:
    if "camp_total_descuelgues" in df.columns:
        col = pd.to_numeric(df["camp_total_descuelgues"], errors="coerce")
        total = len(df)
        gt0 = (col > 0).sum()
        eq0 = (col == 0).sum()
        nulls = col.isna().sum()

        print(f"\nDataset: {nombre}")
        print(f"Total filas                : {total:,}")
        print(f"camp_total_descuelgues > 0 : {gt0:,}")
        print(f"camp_total_descuelgues = 0 : {eq0:,}")
        print(f"camp_total_descuelgues NaN : {nulls:,}")
    else:
        print(f"\nDataset: {nombre}")
        print("No existe la columna 'camp_total_descuelgues'.")

# =========================
# 6) COMPARAR DISTRIBUCIÓN DE COLUMNAS CLAVE
# =========================
print("\n=========================")
print("COLUMNAS IMPORTANTES")
print("=========================")

important_cols = [
    "camp_total_descuelgues",
    "prob_venta_modelo",
    "prob_descuelgue_modelo",
    "ganada"
]

for col in important_cols:
    print(f"\n--- Columna: {col} ---")
    for nombre, df in [("resultados", df_resultados), ("descuelgue", df_descuelgue)]:
        if col in df.columns:
            serie = pd.to_numeric(df[col], errors="coerce")
            print(f"{nombre}:")
            print(f"  no nulos: {serie.notna().sum():,}")
            print(f"  nulos   : {serie.isna().sum():,}")
            if serie.notna().sum() > 0:
                print(f"  min     : {serie.min()}")
                print(f"  max     : {serie.max()}")
                print(f"  media   : {serie.mean()}")
        else:
            print(f"{nombre}: no existe")

# =========================
# 7) SI HAY CLAVE, HACER MERGE DE DIAGNÓSTICO
# =========================
print("\n=========================")
print("MERGE DE DIAGNÓSTICO")
print("=========================")

if candidate_keys:
    best_key = candidate_keys[0]
    print(f"Usando clave candidata por defecto: {best_key}")

    left = df_resultados.copy()
    right = df_descuelgue.copy()

    left[best_key] = left[best_key].astype(str).str.strip()
    right[best_key] = right[best_key].astype(str).str.strip()

    merged = left.merge(
        right[[best_key] + [c for c in right.columns if c == "prob_descuelgue_modelo"]],
        on=best_key,
        how="outer",
        indicator=True,
        suffixes=("_resultados", "_descuelgue")
    )

    print(merged["_merge"].value_counts(dropna=False))

    print("\nEjemplos solo en resultados:")
    print(merged.loc[merged["_merge"] == "left_only", [best_key]].head(10))

    print("\nEjemplos solo en descuelgue:")
    print(merged.loc[merged["_merge"] == "right_only", [best_key]].head(10))

    print("\nEjemplos emparejados:")
    cols_show = [best_key]
    if "prob_descuelgue_modelo" in merged.columns:
        cols_show.append("prob_descuelgue_modelo")
    print(merged.loc[merged["_merge"] == "both", cols_show].head(10))
else:
    print("No hay clave candidata automática. Necesitas identificar una columna ID común.")

print("\n=========================")
print("CONCLUSIÓN")
print("=========================")
print("Si los datasets no tienen el mismo número de filas, NO debes asignar columnas por posición.")
print("La solución correcta será hacer un merge por una clave común única.")