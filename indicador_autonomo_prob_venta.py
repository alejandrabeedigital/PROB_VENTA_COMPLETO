# indicador_autonomo_prob_venta.py
# -*- coding: utf-8 -*-

import os
import re
import joblib
import numpy as np
import pandas as pd

from typing import List

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

from scipy.sparse import csr_matrix, hstack


# =========================
# CONFIG
# =========================
INPUT_PATH = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_COMPLETO\t_pr_venta_future.csv"

OUTPUT_PRED_PATH = "predicciones_ct_sociedad_22_umbral_090_010_regla_forma_juridica_prob_venta.csv"
MODEL_PATH = "modelo_autonomo_calibrado_big_prob_venta.joblib"

CHUNK_SIZE = 300_000
N_FEATURES_TEXT = 2**20
RANDOM_SEED = 42

CALIBRATION_MAX_ROWS = 150_000

THRESH_AUTONOMO = 0.90
THRESH_NO_AUTONOMO = 0.10

COLS_NEEDED = [
    "co_cliente",
    "nombre_empresa",
    "no_comer",
    "email",
    "tx_actvad",
    "telefono",
    "website",
    "ct_sociedad",
]


# =========================
# FEATURES (vectorizadas)
# =========================
GENERIC_EMAIL_DOMAINS = {
    "gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "icloud.com", "live.com",
    "hotmail.es", "outlook.es", "yahoo.es", "gmail.es",
    "msn.com", "proton.me", "protonmail.com", "gmx.com", "gmx.es", "aol.com", "movistar.es",
}

LEGAL_FORMS_CORE = r"(?: " \
    r"S\s*\.?\s*L\s*\.?\s*U\s*\.?|" \
    r"S\s*\.?\s*L\s*\.?\s*L\s*\.?|" \
    r"S\s*\.?\s*L\s*\.?\s*P\s*\.?\s*U\s*\.?|" \
    r"S\s*\.?\s*L\s*\.?\s*P\s*\.?|" \
    r"S\s*\.?\s*L\s*\.?|" \
    r"S\s*\.?\s*A\s*\.?\s*U\s*\.?|" \
    r"S\s*\.?\s*A\s*\.?|" \
    r"S\s*\.?\s*C\s*\.?\s*P\s*\.?|" \
    r"S\s*\.?\s*C\s*\.?|" \
    r"S\s*\.?\s*R\s*\.?\s*L\s*\.?|" \
    r"S\s*\.?\s*C\s*\.?\s*V\s*\.?|" \
    r"S\s*\.?\s*C\s*\.?\s*A\s*\.?|" \
    r"S\s*\.?\s*L\s*\.?\s*N\s*\.?\s*E\s*\.?|" \
    r"C\s*\.?\s*B\s*\.?|" \
    r"C\s*\.?\s*V\s*\.?|" \
    r"COOP\s*\.?|" \
    r"SOCIEDAD|" \
    r"ASOCIACI[ÓO]N|" \
    r"FUNDACI[ÓO]N|" \
    r"UTE|" \
    r"A\.?\s*I\.?\s*E\.?|" \
    r"A\.?\s*E\.?\s*I\.?\s*E\.?|" \
    r"SAT" \
    r")"

SEP_L = r"(?:^|[\s\(\[\{,.;:/\-])"
SEP_R = r"(?=$|[\s\)\]\},.;:/\-])"

LEGAL_FORMS_RE = re.compile(SEP_L + LEGAL_FORMS_CORE + SEP_R, flags=re.IGNORECASE | re.VERBOSE)

NON_PERSON_TOKENS_RE = re.compile(
    r"\b(?:"
    r"AYUNTAMIENTO|INSTITUTO|COLEGIO|PARROQUIA|FEDERACI[ÓO]N|ASOCIACI[ÓO]N|FUNDACI[ÓO]N|"
    r"CLUB|SERVICIO|CENTRO|COMUNIDAD|HOSPITAL|UNIVERSIDAD|COOPERATIVA|CONSORCIO|"
    r"MINISTERIO|CONSEJER(?:IA|ÍA)|DIPUTACI(?:ON|ÓN)|GENERALITAT|GOBIERNO|CABILDO|"
    r"MUSEO|TEATRO|BIBLIOTECA|RESIDENCIA|COLEGIO\s+OFICIAL|COFRAD(?:IA|ÍA)|HERMANDAD|"
    r"ADMINISTRACI(?:ON|ÓN)|ADMINISTRADOR|AUTORIDAD|PORTUARIA|PAGADUR(?:IA|ÍA)|"
    r"ENTIDAD|P[ÚU]BLICA|EMPRESARIAL|INFRAESTRUCTURAS|FERROVIARIAS|"
    r"GERENCIA|TESORER(?:IA|ÍA)|TESORERIA|SEGURIDAD|SOCIAL|TGSS|"
    r"CRUZ|ROJA|ADIF|"
    r"SALUD|SAUDE|SANITARI[AO]S?|"
    r"AJUNTAMENT|CONCELLO|XUNTA|FUNDACIO|ASSOCIACIO|ESCOLA|ELKARTEA|"
    r"JUNTA|ANDALUC(?:IA|ÍA)|GALICIA|ASTURIAS|PRINCIPADO|CATALUNYA|ESPA[NÑ]A|"
    r"PARTIDO|SOCIALISTA|OBRERO|POPULAR|COMISIONES|OBRERAS|"
    r"AGRUPACI[ÓO]N|DEPORTIVA|ESPORTISTES|SOLIDARIS|"
    r"UTE|IES|MATEPSS|CP|CDAD|PROP"
    r")\b",
    flags=re.IGNORECASE
)

NON_PERSON_PHRASES_RE = re.compile(
    r"\b(?:"
    r"CRUZ\s+ROJA|"
    r"SEGURIDAD\s+SOCIAL|"
    r"TESORER(?:IA|ÍA)\s+GENERAL|"
    r"TESORERIA\s+GENERAL|"
    r"GERENCIA\s+DE\s+INFORM[ÁA]TICA|"
    r"SERVICIO\s+(?:CANARIO|GALEGO|DE\s+SALUD)\s+DE\s+SALUD|"
    r"SERVICIO\s+GALEGO\s+DA\s+SAUDE|"
    r"SERVICIO\s+DE\s+SALUD\s+DE\s+CASTILLA\s+LA\s+MANCHA|"
    r"CENTRO\S?\s+DE\s+SALUD|"
    r"ADMINISTRADOR(?:A)?\s+DE\s+INFRAESTRUCTURAS\s+FERROVIARIAS|"
    r"INFRAESTRUCTURAS\s+FERROVIARIAS|"
    r"ESTACION\s+DE\s+TRENES|"
    r"COFRADIA\s+DE\s+PESCADORES|"
    r"AGRUPACI[ÓO]N\s+DEPORTIVA|"
    r"PARTIDO\s+SOCIALISTA|"
    r"PARTIDO\s+POPULAR|"
    r"ENTIDAD\s+P[ÚU]BLICA|"
    r"P[ÚU]BLICA\s+EMPRESARIAL|"
    r"ADMINISTRADOR\s+INFRAESTRUCTURAS|"
    r"COMISIONES\s+OBRERAS|"
    r"SOCIALISTA\s+OBRERO|"
    r"OBRERO\s+ESPA[NÑ]OL|"
    r"CAJA\s+RURAL|"
    r"COM\s+PROP"
    r")\b",
    flags=re.IGNORECASE
)


def normalize_text_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def normalize_ct_sociedad(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()
    s = s.str.replace('"', "", regex=False)
    s = s.str.replace(".0", "", regex=False)
    is_digit = s.str.match(r"^\d+$", na=False)
    s.loc[is_digit] = s.loc[is_digit].str.zfill(2)
    return s


def join_text_fields(df: pd.DataFrame) -> pd.Series:
    ne = normalize_text_series(df["nombre_empresa"])
    nc = normalize_text_series(df["no_comer"])
    act = normalize_text_series(df["tx_actvad"])
    return (ne + " | " + nc + " | " + act).str.strip()


PERSONA_STOPWORDS = {
    "DE", "DEL", "LA", "LAS", "LOS", "Y", "E", "DA", "DO", "DOS", "SAN", "SANTA",
    "DON", "DONA", "SR", "SRA"
}

ACTIVIDAD_INSTITUCIONAL_RE = re.compile(
    r"\b(?:ORGANISMOS\s+OFICIALES|PARROQUIAS|COLEGIOS|INSTITUTOS|UNIVERSIDADES|AYUNTAMIENTOS|EMBAJADAS)\b",
    flags=re.IGNORECASE
)


def build_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    ne = normalize_text_series(df["nombre_empresa"])
    nc = normalize_text_series(df["no_comer"])
    act = normalize_text_series(df["tx_actvad"])
    em = normalize_text_series(df["email"])
    tel = normalize_text_series(df["telefono"])
    web = normalize_text_series(df["website"])

    has_legal_ne = ne.str.contains(LEGAL_FORMS_RE, regex=True, na=False)
    has_legal_nc = nc.str.contains(LEGAL_FORMS_RE, regex=True, na=False)
    out["tiene_forma_juridica"] = (has_legal_ne | has_legal_nc).astype(np.int8)

    np_ne = ne.str.contains(NON_PERSON_TOKENS_RE, regex=True, na=False) | ne.str.contains(NON_PERSON_PHRASES_RE, regex=True, na=False)
    np_nc = nc.str.contains(NON_PERSON_TOKENS_RE, regex=True, na=False) | nc.str.contains(NON_PERSON_PHRASES_RE, regex=True, na=False)
    np_act = act.str.contains(NON_PERSON_PHRASES_RE, regex=True, na=False) | act.str.contains(ACTIVIDAD_INSTITUCIONAL_RE, regex=True, na=False)

    non_person_any = (np_ne | np_nc | np_act)
    out["tiene_token_no_persona"] = non_person_any.astype(np.int8)

    ne_clean = ne.fillna("").astype(str).str.strip()
    has_digit = ne_clean.str.contains(r"\d", regex=True, na=False)
    parts = ne_clean.str.upper().str.split()

    token_count = parts.apply(lambda xs: sum(1 for t in xs if t and t not in PERSONA_STOPWORDS))

    def _bad_lengths(xs):
        xs2 = [t for t in xs if t and t not in PERSONA_STOPWORDS]
        if not xs2:
            return True
        lens = [len(t) for t in xs2]
        if max(lens) > 18:
            return True
        if sum(1 for l in lens if l <= 2) >= 2:
            return True
        return False

    bad_len = parts.apply(_bad_lengths)
    actividad_institucional = act.str.contains(ACTIVIDAD_INSTITUCIONAL_RE, regex=True, na=False)

    looks_person = (
        token_count.between(2, 4)
        & (~out["tiene_forma_juridica"].astype(bool))
        & (~non_person_any)
        & (~has_digit)
        & (~bad_len)
        & (~actividad_institucional)
    )
    out["parece_nombre_persona"] = looks_person.astype(np.int8)

    em_lower = em.str.lower()
    has_at = em_lower.str.contains("@", na=False)
    dom = em_lower.str.split("@").str[-1]
    dom = dom.where(has_at, "")
    out["email_tiene"] = (em != "").astype(np.int8)
    out["email_dominio_generico"] = dom.isin(GENERIC_EMAIL_DOMAINS).astype(np.int8)
    out["email_dominio_corporativo"] = ((dom != "") & (~dom.isin(GENERIC_EMAIL_DOMAINS))).astype(np.int8)

    digits = tel.str.replace(r"\D+", "", regex=True)
    digits = digits.str.replace(r"^34", "", regex=True)
    out["telefono_tiene"] = (digits != "").astype(np.int8)
    out["telefono_es_movil"] = digits.str.match(r"^[67]").fillna(False).astype(np.int8)
    out["telefono_es_fijo"] = digits.str.match(r"^[89]").fillna(False).astype(np.int8)

    out["website_tiene"] = (web != "").astype(np.int8)

    out["n_campos_informados"] = (
        (ne != "").astype(np.int8)
        + (nc != "").astype(np.int8)
        + (act != "").astype(np.int8)
        + (em != "").astype(np.int8)
        + (tel != "").astype(np.int8)
        + (web != "").astype(np.int8)
    ).astype(np.int8)

    out["len_nombre_empresa"] = ne.str.len().astype(np.int32)
    out["len_no_comer"] = nc.str.len().astype(np.int32)
    out["len_actividad"] = act.str.len().astype(np.int32)

    return out


def decide_label(p: float,
                 tiene_forma_juridica: int,
                 tiene_token_no_persona: int,
                 email_dominio_corporativo: int,
                 telefono_es_fijo: int,
                 telefono_es_movil: int) -> str:
    if int(tiene_forma_juridica) == 1:
        return "NO_AUTONOMO"
    if int(tiene_token_no_persona) == 1:
        return "NO_AUTONOMO"
    if int(email_dominio_corporativo) == 1 and int(telefono_es_fijo) == 1 and int(telefono_es_movil) == 0:
        return "NO_AUTONOMO"
    if p >= THRESH_AUTONOMO:
        return "00"
    if p <= THRESH_NO_AUTONOMO:
        return "NO_AUTONOMO"
    return "22"


def simple_reasons(df_rules: pd.DataFrame) -> pd.Series:
    reasons = []
    for col, tag in [
        ("tiene_forma_juridica", "forma_juridica"),
        ("tiene_token_no_persona", "no_persona"),
        ("parece_nombre_persona", "nombre_persona"),
        ("email_dominio_generico", "email_generico"),
        ("email_dominio_corporativo", "email_corporativo"),
        ("telefono_es_movil", "telefono_movil"),
        ("telefono_es_fijo", "telefono_fijo"),
        ("website_tiene", "website"),
    ]:
        if col in df_rules.columns:
            reasons.append(np.where(df_rules[col].astype(bool), tag, ""))
        else:
            reasons.append(np.array([""] * len(df_rules), dtype=object))

    stacked = np.vstack(reasons).T
    out = []
    for row in stacked:
        tags = [t for t in row if t]
        out.append(",".join(tags[:4]))
    return pd.Series(out, index=df_rules.index)


def iter_csv_chunks(path: str, chunksize: int):
    keep = set(COLS_NEEDED)
    for chunk in pd.read_csv(
        path,
        sep=",",
        dtype=str,
        usecols=lambda c: c in keep,
        chunksize=chunksize,
        encoding="utf-8",
        low_memory=False,
    ):
        for c in COLS_NEEDED:
            if c not in chunk.columns:
                chunk[c] = ""

        chunk = chunk.fillna("")
        chunk["ct_sociedad"] = normalize_ct_sociedad(chunk["ct_sociedad"])
        chunk["co_cliente"] = chunk["co_cliente"].fillna("").astype(str).str.strip()
        yield chunk


def iter_csv_chunks_full(path: str, chunksize: int):
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

        for c in COLS_NEEDED:
            if c not in chunk.columns:
                chunk[c] = ""

        chunk["ct_sociedad"] = normalize_ct_sociedad(chunk["ct_sociedad"])
        chunk["co_cliente"] = chunk["co_cliente"].fillna("").astype(str).str.strip()
        yield chunk


def stable_split_mask(groups: pd.Series, test_mod: int = 5, test_bucket: int = 0) -> pd.Series:
    import zlib

    def crc32_int(x: str) -> int:
        return zlib.crc32(x.encode("utf-8")) & 0xffffffff

    buckets = groups.fillna("").astype(str).map(crc32_int) % test_mod
    return buckets == test_bucket


def compute_sample_weight(y: np.ndarray) -> np.ndarray:
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.ones(n, dtype=np.float32)
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


class AutonomoModel:
    def __init__(self):
        self.vec = HashingVectorizer(
            n_features=N_FEATURES_TEXT,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )
        self.clf = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            penalty="l2",
            learning_rate="optimal",
            max_iter=1,
            tol=None,
            average=True,
            random_state=RANDOM_SEED,
        )

        self.calibrated = None
        self.rule_cols = None

    def _make_X(self, df: pd.DataFrame) -> csr_matrix:
        text = join_text_fields(df)
        X_text = self.vec.transform(text)

        rules = build_rule_features(df)
        if self.rule_cols is None:
            self.rule_cols = list(rules.columns)
        else:
            rules = rules[self.rule_cols]

        X_rules = csr_matrix(rules.values)
        return hstack([X_text, X_rules], format="csr")

    def partial_fit(self, df: pd.DataFrame, y: np.ndarray, classes: np.ndarray):
        X = self._make_X(df)
        sw = compute_sample_weight(y)
        self.clf.partial_fit(X, y, classes=classes, sample_weight=sw)

    def fit_calibrator_cv(self, df_cal: pd.DataFrame, y_cal: np.ndarray, cv: int = 3):
        X_cal = self._make_X(df_cal)
        self.calibrated = CalibratedClassifierCV(
            estimator=self.clf,
            method="sigmoid",
            cv=cv,
        )
        self.calibrated.fit(X_cal, y_cal)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._make_X(df)
        if self.calibrated is not None:
            return self.calibrated.predict_proba(X)[:, 1]
        return self.clf.predict_proba(X)[:, 1]


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"No encuentro el archivo: {INPUT_PATH}")

    np.random.seed(RANDOM_SEED)

    input_cols = pd.read_csv(INPUT_PATH, nrows=0, encoding="utf-8").columns.str.strip().tolist()
    pred_cols = input_cols + ["p_autonomo", "reasons_debug", "ct_sociedad_pred"]

    model = AutonomoModel()
    classes = np.array([0, 1], dtype=int)

    cal_rows: List[pd.DataFrame] = []
    cal_y: List[np.ndarray] = []
    cal_count = 0

    seen = 0
    seen_pos = 0

    print("== 1) ENTRENAMIENTO EN STREAMING (ct_sociedad != '22') ==")

    for i, chunk in enumerate(iter_csv_chunks(INPUT_PATH, CHUNK_SIZE), start=1):
        labeled = chunk[chunk["ct_sociedad"] != "22"].copy()
        if labeled.empty:
            if i % 10 == 0:
                print(f"[chunk {i}] sin etiquetados.")
            continue

        y_all = (labeled["ct_sociedad"] == "00").astype(int).to_numpy()

        is_cal = stable_split_mask(labeled["co_cliente"], test_mod=5, test_bucket=0)

        train_df = labeled[~is_cal]
        train_y = y_all[~is_cal.to_numpy()]

        cal_df = labeled[is_cal]
        cal_y_part = y_all[is_cal.to_numpy()]

        if len(train_df) > 0:
            model.partial_fit(train_df, train_y, classes=classes)

        if len(cal_df) > 0 and cal_count < CALIBRATION_MAX_ROWS:
            space_left = CALIBRATION_MAX_ROWS - cal_count
            if len(cal_df) > space_left:
                cal_df = cal_df.iloc[:space_left]
                cal_y_part = cal_y_part[:space_left]
            cal_rows.append(cal_df)
            cal_y.append(cal_y_part)
            cal_count += len(cal_df)

        seen += len(labeled)
        seen_pos += int(y_all.sum())

        if i % 5 == 0:
            print(f"[chunk {i}] etiquetados vistos: {seen:,} | autónomos: {seen_pos:,} | cal: {cal_count:,}")

    df_cal = pd.concat(cal_rows, axis=0, ignore_index=True) if cal_rows else pd.DataFrame(columns=COLS_NEEDED)
    y_cal = np.concatenate(cal_y) if cal_y else np.array([], dtype=int)

    print("\n== 2) CALIBRACIÓN DE PROBABILIDADES ==")
    if len(df_cal) > 0:
        model.fit_calibrator_cv(df_cal, y_cal, cv=3)
        p_cal = model.predict_proba(df_cal)
        try:
            print("ROC AUC (cal):", round(roc_auc_score(y_cal, p_cal), 4))
            print("AUC-PR (cal):", round(average_precision_score(y_cal, p_cal), 4))
            print("Report (umbral 0.5) en cal:")
            print(classification_report(y_cal, (p_cal >= 0.5).astype(int), digits=4))
        except Exception:
            pass
    else:
        print("No hay calibración; probas menos fiables (no recomendado).")

    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")

    print("\n== 3) PREDICCIÓN EN STREAMING PARA ct_sociedad == '22' ==")

    if os.path.exists(OUTPUT_PRED_PATH):
        os.remove(OUTPUT_PRED_PATH)

    wrote_header = False
    total_22 = 0

    for i, chunk_full in enumerate(iter_csv_chunks_full(INPUT_PATH, CHUNK_SIZE), start=1):
        mask_22 = chunk_full["ct_sociedad"] == "22"
        if not mask_22.any():
            if i % 10 == 0:
                print(f"[chunk {i}] sin '22'.")
            continue

        unk_full = chunk_full.loc[mask_22].copy()
        unk = unk_full[COLS_NEEDED].copy()

        p = model.predict_proba(unk)
        unk_rules = build_rule_features(unk)

        out = unk_full.copy()
        out["p_autonomo"] = p
        out["reasons_debug"] = simple_reasons(unk_rules)

        out["ct_sociedad_pred"] = [
            decide_label(pv, fj, npj, edc, tf, tm)
            for pv, fj, npj, edc, tf, tm in zip(
                out["p_autonomo"].to_numpy(),
                unk_rules["tiene_forma_juridica"].to_numpy(),
                unk_rules["tiene_token_no_persona"].to_numpy(),
                unk_rules["email_dominio_corporativo"].to_numpy(),
                unk_rules["telefono_es_fijo"].to_numpy(),
                unk_rules["telefono_es_movil"].to_numpy(),
            )
        ]

        for c in pred_cols:
            if c not in out.columns:
                out[c] = ""

        out = out[pred_cols].copy()

        out.to_csv(
            OUTPUT_PRED_PATH,
            sep=";",
            index=False,
            encoding="utf-8",
            mode="a",
            header=(not wrote_header),
        )
        wrote_header = True

        total_22 += len(out)
        print(f"[chunk {i}] '22' procesados: {len(out):,} | acumulado: {total_22:,}")

    print("\n== FIN ==")
    print(f"Predicciones guardadas en: {OUTPUT_PRED_PATH}")
    print(f"Total '22' predichos: {total_22:,}")
    print(f"Umbrales: AUTÓNOMO >= {THRESH_AUTONOMO} | NO_AUTÓNOMO <= {THRESH_NO_AUTONOMO} | resto -> 22")


if __name__ == "__main__":
    main()