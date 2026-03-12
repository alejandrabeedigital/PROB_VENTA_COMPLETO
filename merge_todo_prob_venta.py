# merge_todo.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import joblib

# Import necesario para que joblib/pickle resuelva la clase
from indicador_autonomo_prob_venta import AutonomoModel, build_rule_features  # noqa: F401


# =========================
# PATHS / CONFIG
# =========================
PROJECT = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_COMPLETO"
MODEL_PATH = os.path.join(PROJECT, "modelo_autonomo_calibrado_big_prob_venta.joblib")
INPUT_PATH = os.path.join(PROJECT, "t_pr_venta_future.csv")

# CSV de 22 ya calculado
PRED_22_PATH = os.path.join(
    PROJECT,
    "predicciones_ct_sociedad_22_umbral_090_010_regla_forma_juridica_prob_venta.csv"
)

OUT_PATH = os.path.join(PROJECT, "t_pr_venta_con_pred_todo.csv")

TH_AUT = 0.90
TH_NO = 0.10
CHUNK_SIZE = 300_000

COLS_MIN = [
    "co_cliente",
    "nombre_empresa",
    "no_comer",
    "email",
    "tx_actvad",
    "telefono",
    "website",
    "ct_sociedad",
]

NEW_COLS = [
    "p_autonomo",
    "ct_sociedad_pred",
    "outcome_pred",
    "outcome_sin_con_pred",
]


def normalize_ct_sociedad_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()

    # quitar comillas y sufijo .0 típico de floats
    s = s.str.replace('"', "", regex=False)
    s = s.str.replace(".0", "", regex=False)

    is_digit = s.str.match(r"^\d+$", na=False)
    s.loc[is_digit] = s.loc[is_digit].str.zfill(2)
    return s


def iter_csv_chunks_full(path: str, chunksize: int):
    """
    Lee en streaming TODAS las columnas del CSV de entrada.
    Para t_pr_venta el separador es coma.
    """
    for chunk in pd.read_csv(
        path,
        sep=",",
        dtype=str,
        chunksize=chunksize,
        encoding="utf-8",
        low_memory=False,
    ):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.fillna("")

        # Garantizar columnas mínimas aunque falten
        for c in COLS_MIN:
            if c not in chunk.columns:
                chunk[c] = ""

        chunk["ct_sociedad"] = normalize_ct_sociedad_series(chunk["ct_sociedad"])
        chunk["co_cliente"] = chunk["co_cliente"].fillna("").astype(str).str.strip()
        yield chunk


def decide_pred_from_p(p: np.ndarray, th_aut: float, th_no: float) -> np.ndarray:
    """
    Devuelve:
      1 = autónomo seguro
      0 = no autónomo seguro
     -1 = zona gris
    """
    pred = np.full(len(p), -1, dtype=np.int8)
    pred[p >= th_aut] = 1
    pred[p <= th_no] = 0
    return pred


def outcome_from_ct_pred(ct_pred: pd.Series) -> pd.Series:
    ct = ct_pred.fillna("").astype(str)
    out = np.full(len(ct), "DESCONOCIDO", dtype=object)
    out[ct.to_numpy() == "00"] = "AUTONOMO"
    out[ct.to_numpy() == "NO_AUTONOMO"] = "NO_AUTONOMO"
    return pd.Series(out, index=ct.index)


def ensure_output_columns(df: pd.DataFrame, final_cols: list) -> pd.DataFrame:
    """
    Mantiene TODAS las columnas originales y garantiza que existan
    las nuevas columnas añadidas, devolviendo exactamente el mismo
    orden de columnas en ambos bloques.
    """
    for c in final_cols:
        if c not in df.columns:
            df[c] = ""

    if "ct_sociedad" in df.columns:
        df["ct_sociedad"] = normalize_ct_sociedad_series(df["ct_sociedad"])

    if "co_cliente" in df.columns:
        df["co_cliente"] = df["co_cliente"].fillna("").astype(str).str.strip()

    return df[final_cols].copy()


def build_base_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el DF mínimo que necesita el modelo/rules a partir
    del chunk completo.
    """
    base = df.copy()

    for c in COLS_MIN:
        if c not in base.columns:
            base[c] = ""

    base = base.fillna("")
    base["ct_sociedad"] = normalize_ct_sociedad_series(base["ct_sociedad"])
    base["co_cliente"] = base["co_cliente"].fillna("").astype(str).str.strip()
    return base


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No encuentro el modelo en: {MODEL_PATH}")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"No encuentro el input en: {INPUT_PATH}")
    if not os.path.exists(PRED_22_PATH):
        raise FileNotFoundError(f"No encuentro el CSV de 22 ya calculado en: {PRED_22_PATH}")

    input_cols = pd.read_csv(INPUT_PATH, nrows=0, encoding="utf-8").columns.str.strip().tolist()
    final_cols = input_cols + [c for c in NEW_COLS if c not in input_cols]

    model = joblib.load(MODEL_PATH)
    print("Modelo cargado OK:", MODEL_PATH)
    print("Usando 22 ya calculado:", PRED_22_PATH)
    print("Output final:", OUT_PATH)

    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    wrote_header = False

    # =========================
    # 1) ETIQUETADOS (ct != 22)
    # =========================
    print("\n== Generando predicciones para ETIQUETADOS (ct != '22') en streaming ==")

    total_et = 0
    for i, chunk_full in enumerate(iter_csv_chunks_full(INPUT_PATH, CHUNK_SIZE), start=1):
        chunk_model = build_base_for_model(chunk_full)

        mask_lab = chunk_model["ct_sociedad"] != "22"
        if not mask_lab.any():
            continue

        lab_model = chunk_model.loc[mask_lab].copy()
        lab_full = chunk_full.loc[mask_lab].copy()

        p = np.asarray(model.predict_proba(lab_model), dtype=np.float32)
        pred = decide_pred_from_p(p, TH_AUT, TH_NO)

        rules = build_rule_features(lab_model)
        fj = rules["tiene_forma_juridica"].to_numpy(dtype=np.int8)
        npj = rules["tiene_token_no_persona"].to_numpy(dtype=np.int8)
        edc = rules["email_dominio_corporativo"].to_numpy(dtype=np.int8)
        tf = rules["telefono_es_fijo"].to_numpy(dtype=np.int8)
        tm = rules["telefono_es_movil"].to_numpy(dtype=np.int8)

        institucional = (edc == 1) & (tf == 1) & (tm == 0)

        ct_pred = np.full(len(lab_model), "22", dtype=object)
        force_no = (fj == 1) | (npj == 1) | institucional
        ct_pred[force_no] = "NO_AUTONOMO"

        free = ~force_no
        ct_pred[free & (pred == 1)] = "00"
        ct_pred[free & (pred == 0)] = "NO_AUTONOMO"

        out = lab_full.copy()
        out["p_autonomo"] = p
        out["ct_sociedad_pred"] = ct_pred
        out["outcome_pred"] = outcome_from_ct_pred(out["ct_sociedad_pred"])

        # Aquí usamos la verdad conocida
        out["outcome_sin_con_pred"] = np.where(
            lab_model["ct_sociedad"] == "00",
            "AUTONOMO",
            "NO_AUTONOMO"
        )

        out_final = ensure_output_columns(out, final_cols)

        out_final.to_csv(
            OUT_PATH,
            sep=",",
            index=False,
            encoding="utf-8",
            mode="a",
            header=(not wrote_header),
        )
        wrote_header = True

        total_et += len(out_final)
        if i % 5 == 0:
            print(f"[chunk {i}] etiquetados añadidos: {total_et:,}".replace(",", "."))

    print(f"OK etiquetados: {total_et:,}".replace(",", "."))

    # =========================
    # 2) 22 YA CALCULADO
    # =========================
    print("\n== Añadiendo ct == '22' desde CSV ya calculado (sin recalcular) ==")

    total_22 = 0
    for j, df22 in enumerate(
        pd.read_csv(
            PRED_22_PATH,
            sep=";",
            dtype=str,
            chunksize=CHUNK_SIZE,
            encoding="utf-8",
            low_memory=False,
        ),
        start=1
    ):
        df22.columns = df22.columns.str.strip()
        df22 = df22.fillna("")

        if "ct_sociedad_pred" not in df22.columns:
            raise ValueError("El CSV de 22 no tiene columna 'ct_sociedad_pred'. Revisa PRED_22_PATH.")

        if "ct_sociedad" in df22.columns:
            df22["ct_sociedad"] = normalize_ct_sociedad_series(df22["ct_sociedad"])

        if "co_cliente" in df22.columns:
            df22["co_cliente"] = df22["co_cliente"].fillna("").astype(str).str.strip()

        df22["outcome_pred"] = outcome_from_ct_pred(df22["ct_sociedad_pred"])

        # Para los 22, aquí sí usamos la predicción
        df22["outcome_sin_con_pred"] = df22["outcome_pred"]

        # Si no trae p_autonomo, la creamos vacía
        if "p_autonomo" not in df22.columns:
            df22["p_autonomo"] = ""

        df22_final = ensure_output_columns(df22, final_cols)

        df22_final.to_csv(
            OUT_PATH,
            sep=",",
            index=False,
            encoding="utf-8",
            mode="a",
            header=False,
        )

        total_22 += len(df22_final)
        if j % 5 == 0:
            print(f"[chunk22 {j}] 22 añadidos: {total_22:,}".replace(",", "."))

    print(f"OK 22: {total_22:,}".replace(",", "."))

    print("\n== FIN ==")
    print("Guardado:", OUT_PATH)
    print("NOTA: el fichero final conserva todas las columnas originales y añade:")
    print(";".join(NEW_COLS))
    print("NOTA 2: el fichero final concatena primero etiquetados y luego 22 (no mantiene el orden original fila a fila).")


if __name__ == "__main__":
    main()