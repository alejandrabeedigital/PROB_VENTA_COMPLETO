import pandas as pd
import numpy as np

ARCHIVO_IN = "t_pr_venta_future.csv"
ARCHIVO_OUT = "nuevasvars1.csv"

def normalize_bool_series(s: pd.Series) -> pd.Series:
    """
    Convierte valores típicos a booleano pandas (True/False) o NaN:
    - True/False
    - 1/0
    - "True"/"False"
    - "1"/"0"
    - "SI"/"NO", "S"/"N"
    """
    if s is None:
        return pd.Series([pd.NA])

    # Si ya es booleano, devolvemos como boolean con NA
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")

    # Normalizamos strings
    s2 = s.copy()

    # Pasar a string donde no sea NaN
    s2 = s2.astype("string")

    s2 = s2.str.strip().str.lower()

    true_set = {"true", "1", "si", "sí", "s", "y", "yes"}
    false_set = {"false", "0", "no", "n"}

    out = pd.Series(pd.NA, index=s2.index, dtype="boolean")
    out[s2.isin(true_set)] = True
    out[s2.isin(false_set)] = False

    return out

def main():
    df = pd.read_csv(ARCHIVO_IN, low_memory=False)
    print(f"Filas al cargar: {len(df):,}")

    # -------------------------
    # Asegurar columnas existen
    # -------------------------
    needed = ["excliente", "dias_desde_ult_cont", "ranking_number", "claim_business", "website", "website"]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas en el CSV: {missing_cols}")

    # -------------------------
    # Normalizaciones de tipos
    # -------------------------
    excliente_bool = normalize_bool_series(df["excliente"])   # boolean con NA
    df["dias_desde_ult_cont"] = pd.to_numeric(df["dias_desde_ult_cont"], errors="coerce")
    df["ranking_number"] = pd.to_numeric(df["ranking_number"], errors="coerce")
    claim_num = pd.to_numeric(df["claim_business"], errors="coerce")  # puede ser 0/1 o NaN

    # Convertimos claim_business a booleano cuando se pueda:
    claim_bool = pd.Series(pd.NA, index=df.index, dtype="boolean")
    claim_bool[claim_num == 1] = True
    claim_bool[claim_num == 0] = False

    # -------------------------
    # 1) excliente_cat
    # excliente_cat = case_when(
    #   !excliente ~ "0 .No_excliente",
    #   dias_desde_ult_cont < 360 ~ "excliente_reciente",
    #   dias_desde_ult_cont >= 360 ~ "excliente_antiguo",
    #   TRUE ~ NA
    # )
    # -------------------------
    df["excliente_cat"] = pd.NA

    # !excliente  -> No excliente
    df.loc[excliente_bool == False, "excliente_cat"] = "0 .No_excliente"

    # excliente True & dias < 360
    df.loc[(excliente_bool == True) & (df["dias_desde_ult_cont"] < 360), "excliente_cat"] = "excliente_reciente"

    # excliente True & dias >= 360
    df.loc[(excliente_bool == True) & (df["dias_desde_ult_cont"] >= 360), "excliente_cat"] = "excliente_antiguo"

    # El resto se queda en NA (por ejemplo: excliente True pero dias_desde_ult_cont NA)

    # -------------------------
    # 2) con_web = !is.na(website) & !(website %in% c("", " "))
    # -------------------------

    # combinar website_x y website_y en una sola serie
    website = df["website"]

    # limpiar espacios
    w = website.astype("string")
    w_stripped = w.str.strip()

    # crear variable
    df["con_web"] = (~w.isna()) & (w_stripped != "")

    # -------------------------
    # 3) sin_gmb = is.na(ranking_number) & is.na(claim_business)
    # -------------------------
    df["sin_gmb"] = df["ranking_number"].isna() & claim_num.isna()

    # -------------------------
    # 4) gmb_sin_owner = !is.na(claim_business) & claim_business
    # OJO: esto en realidad es "tiene ficha y está reclamada" (owner), no "sin owner".
    # Pero lo dejo EXACTO a tu condición.
    # -------------------------
    df["gmb_sin_owner"] = (~claim_num.isna()) & (claim_num == 1)

    # -------------------------
    # 5) ranking_number_cat = case_when(
    #   ranking_number < 10 ~ "bueno",
    #   ranking_number < 15 ~ "medio",
    #   TRUE ~ "z_malo"
    # )
    # Si ranking_number es NaN, cae en TRUE -> "z_malo" en tu lógica R.
    # Si quieres que NaN sea otra categoría, dímelo y lo cambio.
    # -------------------------
    df["ranking_number_cat"] = "z_malo"
    df.loc[df["ranking_number"] < 15, "ranking_number_cat"] = "medio"
    df.loc[df["ranking_number"] < 10, "ranking_number_cat"] = "bueno"

    # -------------------------
    # Checks rápidos
    # -------------------------
    print("\n--- NA counts (nuevas variables) ---")
    for c in ["excliente_cat", "con_web", "sin_gmb", "gmb_sin_owner", "ranking_number_cat"]:
        print(f"{c}: {df[c].isna().sum():,} NaN")

    print("\n--- Distribuciones (top) ---")
    print("\nexcliente_cat:")
    print(df["excliente_cat"].value_counts(dropna=False).head(10))

    print("\ncon_web:")
    print(df["con_web"].value_counts(dropna=False))

    print("\nsin_gmb:")
    print(df["sin_gmb"].value_counts(dropna=False))

    print("\ngmb_sin_owner:")
    print(df["gmb_sin_owner"].value_counts(dropna=False))

    print("\nranking_number_cat:")
    print(df["ranking_number_cat"].value_counts(dropna=False))

    # -------------------------
    # Guardar
    # -------------------------
    df.to_csv(ARCHIVO_OUT, index=False, encoding="utf-8")
    print(f"\n✅ Guardado: {ARCHIVO_OUT}")
    print(f"Filas guardadas: {len(df):,}")

if __name__ == "__main__":
    main()