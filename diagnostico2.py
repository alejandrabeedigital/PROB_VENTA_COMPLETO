import pandas as pd

CSV_RESULTADOS = "todo_con_resultados_17.csv"
CSV_DESCUELGUE = "todo_con_prob_descuelgue.csv"

print("Leyendo CSVs...")
df_resultados = pd.read_csv(CSV_RESULTADOS, low_memory=False)
df_descuelgue = pd.read_csv(CSV_DESCUELGUE, low_memory=False)

# Nos quedamos con el mismo universo que resultados
df_desc_f = df_descuelgue[df_descuelgue["camp_total_descuelgues"] > 0].copy()

print("\n=========================")
print("TAMAÑOS")
print("=========================")
print(f"resultados: {len(df_resultados):,}")
print(f"descuelgue filtrado: {len(df_desc_f):,}")

# Normalizar co_cliente para evitar falsas diferencias
df_resultados["co_cliente"] = df_resultados["co_cliente"].astype(str).str.strip()
df_desc_f["co_cliente"] = df_desc_f["co_cliente"].astype(str).str.strip()

# =========================
# 1) CUÁNTOS DUPLICADOS HAY
# =========================
dup_res_mask = df_resultados["co_cliente"].duplicated(keep=False)
dup_des_mask = df_desc_f["co_cliente"].duplicated(keep=False)

df_res_dup = df_resultados[dup_res_mask].copy()
df_des_dup = df_desc_f[dup_des_mask].copy()

print("\n=========================")
print("DUPLICADOS co_cliente")
print("=========================")
print(f"Filas duplicadas en resultados: {len(df_res_dup):,}")
print(f"Filas duplicadas en descuelgue: {len(df_des_dup):,}")
print(f"Clientes duplicados en resultados: {df_res_dup['co_cliente'].nunique():,}")
print(f"Clientes duplicados en descuelgue: {df_des_dup['co_cliente'].nunique():,}")

# =========================
# 2) TOP CLIENTES MÁS REPETIDOS
# =========================
print("\n=========================")
print("TOP co_cliente MÁS REPETIDOS")
print("=========================")

top_res = df_res_dup["co_cliente"].value_counts().head(20)
top_des = df_des_dup["co_cliente"].value_counts().head(20)

print("\nResultados:")
print(top_res)

print("\nDescuelgue:")
print(top_des)

# =========================
# 3) QUÉ COLUMNAS CAMBIAN ENTRE DUPLICADOS
# =========================
# Columnas que suelen explicar varias filas por cliente
candidate_cols = [
    "id_opp",
    "telefono",
    "campaña",
    "fe_carga_efectiva",
    "lote",
    "movil",
    "plataforma_destino",
    "ganada",
    "camp_total_descuelgues",
    "total_descuelgues",
    "primer_descuelgue_registrado",
    "ult_descuelgue_registrado",
]

candidate_cols = [c for c in candidate_cols if c in df_resultados.columns]

print("\n=========================")
print("ANÁLISIS DE COLUMNAS QUE VARÍAN EN DUPLICADOS")
print("=========================")

def analizar_variacion(df, nombre):
    solo_dup = df[df["co_cliente"].duplicated(keep=False)].copy()
    resumen = []

    for col in candidate_cols:
        nunique_por_cliente = solo_dup.groupby("co_cliente")[col].nunique(dropna=False)
        clientes_con_variacion = (nunique_por_cliente > 1).sum()
        resumen.append({
            "columna": col,
            "clientes_duplicados_con_valores_distintos": int(clientes_con_variacion),
            "max_valores_distintos_en_un_cliente": int(nunique_por_cliente.max())
        })

    resumen_df = pd.DataFrame(resumen).sort_values(
        ["clientes_duplicados_con_valores_distintos", "max_valores_distintos_en_un_cliente"],
        ascending=False
    )

    print(f"\nDataset: {nombre}")
    print(resumen_df.to_string(index=False))

analizar_variacion(df_resultados, "resultados")
analizar_variacion(df_desc_f, "descuelgue_filtrado")

# =========================
# 4) SACAR EJEMPLOS CONCRETOS
# =========================
print("\n=========================")
print("EJEMPLOS CONCRETOS DE DUPLICADOS")
print("=========================")

ejemplos = df_res_dup["co_cliente"].value_counts().head(10).index.tolist()

cols_to_show = [
    "co_cliente",
    "id_opp",
    "telefono",
    "campaña",
    "fe_carga_efectiva",
    "lote",
    "movil",
    "ganada",
    "camp_total_descuelgues",
    "total_descuelgues",
]
cols_to_show = [c for c in cols_to_show if c in df_resultados.columns]

for cliente in ejemplos[:5]:
    print(f"\n----- RESULTADOS | co_cliente = {cliente} -----")
    print(
        df_resultados[df_resultados["co_cliente"] == cliente][cols_to_show]
        .sort_values(cols_to_show[1:2] if len(cols_to_show) > 1 else cols_to_show)
        .to_string(index=False)
    )

    print(f"\n----- DESCUELGUE FILTRADO | co_cliente = {cliente} -----")
    cols_to_show_des = [c for c in cols_to_show if c in df_desc_f.columns]
    print(
        df_desc_f[df_desc_f["co_cliente"] == cliente][cols_to_show_des]
        .sort_values(cols_to_show_des[1:2] if len(cols_to_show_des) > 1 else cols_to_show_des)
        .to_string(index=False)
    )

# =========================
# 5) VER SI SON DUPLICADOS EXACTOS O CASI
# =========================
print("\n=========================")
print("DUPLICADOS EXACTOS")
print("=========================")

exact_dup_res = df_resultados.duplicated().sum()
exact_dup_des = df_desc_f.duplicated().sum()

print(f"Duplicados exactos en resultados: {exact_dup_res:,}")
print(f"Duplicados exactos en descuelgue_filtrado: {exact_dup_des:,}")

# =========================
# 6) COMPROBAR SI co_cliente + id_opp ES ÚNICO
# =========================
print("\n=========================")
print("PRUEBA DE CLAVES COMPUESTAS")
print("=========================")

keys_to_test = [
    ["co_cliente", "id_opp"],
    ["co_cliente", "telefono"],
    ["co_cliente", "id_opp", "telefono"],
    ["co_cliente", "campaña"],
    ["co_cliente", "fe_carga_efectiva"],
    ["co_cliente", "id_opp", "campaña"],
]

for key in keys_to_test:
    if all(c in df_resultados.columns for c in key):
        dup_res = df_resultados.duplicated(subset=key).sum()
        dup_des = df_desc_f.duplicated(subset=key).sum()
        print(f"{key} -> duplicados resultados: {dup_res:,} | duplicados descuelgue: {dup_des:,}")

# =========================
# 7) EXPORTAR DUPLICADOS PARA REVISIÓN
# =========================
df_res_dup.to_csv("duplicados_co_cliente_resultados.csv", index=False)
df_des_dup.to_csv("duplicados_co_cliente_descuelgue_filtrado.csv", index=False)

print("\nArchivos generados:")
print("- duplicados_co_cliente_resultados.csv")
print("- duplicados_co_cliente_descuelgue_filtrado.csv")