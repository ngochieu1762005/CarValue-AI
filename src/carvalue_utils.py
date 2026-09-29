from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "brand", "model", "origin", "body_type", "transmission",
    "fuel_type", "exterior_color", "drivetrain"
]
NUMERIC_FEATURES = [
    "car_age", "mileage_km", "mileage_zero_flag", "engine_size_l", "seats"
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "price_vnd"

SIGNATURE_COLS = [
    "brand", "model", "year", "mileage_km", "origin", "body_type",
    "transmission", "engine", "exterior_color", "seats", "drivetrain"
]


def resolve_project_root(start: Optional[Path] = None) -> Path:
    """Find project root whether code runs from root, notebooks/, src/, or elsewhere nearby."""
    start = Path(start or Path.cwd()).resolve()
    candidates = [start, *start.parents]
    for c in candidates:
        if (c / "data" / "bonbanh_usedcar_dataset.csv").exists() and (c / "src").exists():
            return c
    raise FileNotFoundError(
        "Could not locate project root. Expected data/bonbanh_usedcar_dataset.csv and src/."
    )


def extract_price_from_text(text):
    if pd.isna(text):
        return np.nan
    t = str(text).lower().strip().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:tỷ|tỉ)(?:\s*(\d+(?:\.\d+)?)\s*(?:triệu|tr))?", t)
    if m:
        ty = float(m.group(1))
        tr = float(m.group(2)) if m.group(2) else 0.0
        return int(round(ty * 1_000_000_000 + tr * 1_000_000))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:triệu|tr)\b", t)
    if m:
        return int(round(float(m.group(1)) * 1_000_000))
    return np.nan


def strip_price_suffix(text):
    if pd.isna(text):
        return text
    s = str(text).strip()
    s = re.sub(
        r"\s*[-–—]\s*\d+(?:[.,]\d+)?\s*(?:tỷ|tỉ)(?:\s*\d+(?:[.,]\d+)?\s*(?:triệu|tr))?\s*$",
        "", s, flags=re.I,
    )
    s = re.sub(r"\s*[-–—]\s*\d+(?:[.,]\d+)?\s*(?:triệu|tr)\s*$", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def strip_trailing_year(model, year):
    if pd.isna(model):
        return model
    s = str(model).strip()
    if pd.notna(year):
        y = str(int(float(year)))
        s = re.sub(rf"\s+{re.escape(y)}\s*$", "", s).strip()
    return s


def normalize_brand_model(brand, model):
    b = "" if pd.isna(brand) else str(brand).strip()
    m = "" if pd.isna(model) else str(model).strip()
    rules = [
        ("Mercedes", r"^Benz\s+", "Mercedes-Benz"),
        ("Rolls", r"^Royce\s+", "Rolls-Royce"),
        ("Aston", r"^Martin\s+", "Aston Martin"),
        ("Lynk", r"^&\s*Co\s+", "Lynk & Co"),
    ]
    for base_brand, prefix_re, normalized_brand in rules:
        if b.lower() == base_brand.lower():
            b = normalized_brand
            m = re.sub(prefix_re, "", m, flags=re.I).strip()
            break
    if b == "LandRover":
        b = "Land Rover"
    return b, m


def parse_mileage(x):
    if pd.isna(x):
        return np.nan
    nums = re.sub(r"[^\d]", "", str(x))
    return float(nums) if nums else np.nan


def parse_seats(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"(\d+)", str(x))
    return float(m.group(1)) if m else np.nan


def parse_engine(x):
    if pd.isna(x):
        return (np.nan, np.nan)
    s = re.sub(r"\s+", " ", str(x).strip())
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*L", s, flags=re.I)
    engine_size = float(m.group(1).replace(",", ".")) if m else np.nan
    fuel = re.sub(r"\s*\d+(?:[.,]\d+)?\s*L.*$", "", s, flags=re.I).strip()
    return (fuel if fuel else np.nan, engine_size)


def stable_signature(row: pd.Series) -> str:
    vals = ["" if pd.isna(row[c]) else str(row[c]).strip().lower() for c in SIGNATURE_COLS]
    return hashlib.sha1("||".join(vals).encode("utf-8")).hexdigest()[:16]


def clean_raw_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Apply the same repair/normalization logic used in the supplied EDA notebook."""
    clean = df.copy()
    clean["model_raw"] = clean["model"]
    clean["price_original"] = clean["price_vnd"]

    recovered = clean["model_raw"].apply(extract_price_from_text)
    mask = clean["price_vnd"].isna() & recovered.notna()
    clean.loc[mask, "price_vnd"] = recovered[mask]
    clean["price_recovered_flag"] = mask.astype(int)

    clean["model"] = clean.apply(
        lambda r: strip_trailing_year(strip_price_suffix(r["model_raw"]), r["year"]), axis=1
    )
    normalized = clean.apply(lambda r: normalize_brand_model(r["brand"], r["model"]), axis=1)
    clean["brand"] = [x[0] for x in normalized]
    clean["model"] = [x[1] for x in normalized]

    clean["mileage_km_raw"] = clean["mileage_km"]
    clean["mileage_km"] = clean["mileage_km"].apply(parse_mileage)
    clean["mileage_zero_flag"] = (clean["mileage_km"] == 0).astype(int)
    clean.loc[clean["mileage_km"] == 0, "mileage_km"] = np.nan

    clean["seats_raw"] = clean["seats"]
    clean["seats"] = clean["seats"].apply(parse_seats)

    engine_info = clean["engine"].apply(parse_engine)
    clean["fuel_type"] = engine_info.apply(lambda x: x[0])
    clean["engine_size_l"] = engine_info.apply(lambda x: x[1])

    # Electric listings occasionally contain malformed strings such as
    # "Điện 6.0 L". Engine displacement is not meaningful for an EV, so
    # always keep it missing rather than letting that parsing noise reach the model.
    electric_mask = clean["fuel_type"].astype(str).str.strip().str.casefold().eq("điện".casefold())
    clean.loc[electric_mask, "engine_size_l"] = np.nan

    for c in ["transmission", "drivetrain"]:
        clean[c] = clean[c].replace("-", np.nan)

    clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
    clean["price_vnd"] = pd.to_numeric(clean["price_vnd"], errors="coerce")

    reference_year = int(clean["year"].max())
    clean["car_age"] = reference_year - clean["year"]
    clean["annual_km"] = clean["mileage_km"] / clean["car_age"].replace(0, 1)
    clean["location_missing_flag"] = clean["location"].isna().astype(int)

    clean["vehicle_signature"] = clean.apply(stable_signature, axis=1)
    sig_counts = clean["vehicle_signature"].value_counts()
    clean["signature_group_size"] = clean["vehicle_signature"].map(sig_counts)

    # Flag extreme within-model-year target anomalies. This is deliberately
    # conservative: only groups with >=4 listings are checked, and a row is
    # flagged only when its price is below 1/4 or above 4x the group median.
    # The raw/repaired row is retained for audit, but flagged rows are excluded
    # from supervised model training.
    grp = clean.groupby(["brand", "model", "year"], dropna=False)["price_vnd"].transform("median")
    grp_n = clean.groupby(["brand", "model", "year"], dropna=False)["price_vnd"].transform("count")
    price_ratio = clean["price_vnd"] / grp.replace(0, np.nan)
    clean["price_outlier_flag"] = (
        (grp_n >= 4) & ((price_ratio < 0.25) | (price_ratio > 4.0))
    ).fillna(False).astype(int)

    clean["target_valid_flag"] = (
        clean["price_vnd"].notna() & (clean["price_vnd"] > 0) & clean["price_outlier_flag"].eq(0)
    ).astype(int)
    clean["market_segment"] = np.where(
        clean["body_type"].eq("Truck"), "Commercial/Truck", "Passenger/Light vehicle"
    )
    return clean, reference_year


def create_group_split(model_df: pd.DataFrame, test_size: float = 0.20, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    out = model_df.copy().reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(out, groups=out["vehicle_signature"]))
    out["split"] = "train"
    out.loc[test_idx, "split"] = "test"
    overlap = set(out.loc[out["split"] == "train", "vehicle_signature"]) & set(
        out.loc[out["split"] == "test", "vehicle_signature"]
    )
    if overlap:
        raise AssertionError(f"Group leakage detected: {len(overlap)} signatures overlap")
    return out


def ensure_modeling_artifacts(project_root: Path, force: bool = False) -> Tuple[pd.DataFrame, int]:
    project_root = Path(project_root)
    raw_path = project_root / "data" / "bonbanh_usedcar_dataset.csv"
    artifacts = project_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    repaired_path = artifacts / "CarValue_AI_master_repaired.csv"
    model_path = artifacts / "CarValue_AI_model_candidate.csv"
    meta_path = artifacts / "data_metadata.json"

    if model_path.exists() and meta_path.exists() and not force:
        model_df = pd.read_csv(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return model_df, int(meta["reference_year"])

    raw = pd.read_csv(raw_path)
    clean, reference_year = clean_raw_dataset(raw)
    model_df = clean[clean["target_valid_flag"].eq(1)].copy()
    model_df = create_group_split(model_df)

    clean.to_csv(repaired_path, index=False, encoding="utf-8-sig")
    model_df.to_csv(model_path, index=False, encoding="utf-8-sig")

    meta = {
        "raw_rows": int(len(raw)),
        "model_rows": int(len(model_df)),
        "reference_year": int(reference_year),
        "recovered_price_rows": int(clean["price_recovered_flag"].sum()),
        "remaining_missing_target": int(clean[TARGET].isna().sum()),
        "price_outlier_rows_excluded_from_modeling": int(clean["price_outlier_flag"].sum()),
        "unique_vehicle_signatures": int(clean["vehicle_signature"].nunique()),
        "rows_in_repeated_groups": int((clean["signature_group_size"] > 1).sum()),
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return model_df, reference_year


def get_train_test(model_df: pd.DataFrame):
    train = model_df[model_df["split"] == "train"].copy()
    test = model_df[model_df["split"] == "test"].copy()
    return train, test


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in FEATURES:
        if c not in x.columns:
            x[c] = np.nan
    return x[FEATURES].copy()


def _linear_preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


def _tree_preprocessor():
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


@dataclass
class PriceModelBundle:
    name: str
    kind: str
    model: object
    preprocessor: object = None
    log_target: bool = True
    reference_year: int = 2026

    def _catboost_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = prepare_feature_frame(X)
        for c in CATEGORICAL_FEATURES:
            X[c] = X[c].fillna("Unknown").astype(str)
        for c in NUMERIC_FEATURES:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = prepare_feature_frame(X)
        if self.kind == "catboost_native":
            xp = self._catboost_frame(X)
        elif self.preprocessor is not None:
            xp = self.preprocessor.transform(X)
        else:
            xp = X
        pred = np.asarray(self.model.predict(xp), dtype=float)
        if self.log_target:
            pred = np.expm1(pred)
        return np.maximum(pred, 0.0)


def fit_ridge_bundle(train_df: pd.DataFrame, reference_year: int, alpha: float = 10.0) -> PriceModelBundle:
    X = prepare_feature_frame(train_df)
    y = np.log1p(train_df[TARGET].to_numpy(dtype=float))
    prep = _linear_preprocessor()
    xp = prep.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(xp, y)
    return PriceModelBundle("Ridge Regression", "sklearn", model, prep, True, reference_year)


def fit_random_forest_bundle(train_df: pd.DataFrame, reference_year: int, **kwargs) -> PriceModelBundle:
    X = prepare_feature_frame(train_df)
    y = np.log1p(train_df[TARGET].to_numpy(dtype=float))
    prep = _tree_preprocessor()
    xp = prep.fit_transform(X)
    params = dict(
        n_estimators=120,
        max_depth=None,
        min_samples_leaf=1,
        max_features=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    params.update(kwargs)
    model = RandomForestRegressor(**params)
    model.fit(xp, y)
    return PriceModelBundle("Random Forest", "sklearn", model, prep, True, reference_year)


def fit_xgboost_bundle(train_df: pd.DataFrame, reference_year: int, **kwargs) -> PriceModelBundle:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    X = prepare_feature_frame(train_df)
    y = np.log1p(train_df[TARGET].to_numpy(dtype=float))
    prep = _tree_preprocessor()
    xp = prep.fit_transform(X)
    params = dict(
        n_estimators=420,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        tree_method="hist",
    )
    params.update(kwargs)
    model = XGBRegressor(**params)
    model.fit(xp, y, verbose=False)
    return PriceModelBundle("XGBoost", "sklearn", model, prep, True, reference_year)


def fit_catboost_bundle(train_df: pd.DataFrame, reference_year: int, **kwargs) -> PriceModelBundle:
    if CatBoostRegressor is None:
        raise ImportError("catboost is not installed. Run: pip install catboost")
    X = prepare_feature_frame(train_df)
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].fillna("Unknown").astype(str)
    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = np.log1p(train_df[TARGET].to_numpy(dtype=float))
    params = dict(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    params.update(kwargs)
    model = CatBoostRegressor(**params)
    model.fit(X, y, cat_features=CATEGORICAL_FEATURES)
    return PriceModelBundle("CatBoost", "catboost_native", model, None, True, reference_year)


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rmsle = math.sqrt(mean_squared_error(np.log1p(np.maximum(y_true, 0)), np.log1p(np.maximum(y_pred, 0))))
    return {
        "MAE_VND": float(mae),
        "MAE_million_VND": float(mae / 1e6),
        "RMSE_VND": float(rmse),
        "RMSE_million_VND": float(rmse / 1e6),
        "R2": float(r2),
        "RMSLE": float(rmsle),
    }


def median_baseline_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    return np.full(len(test_df), float(train_df[TARGET].median()))


def save_bundle(bundle: PriceModelBundle, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: Path) -> PriceModelBundle:
    return joblib.load(path)


def train_price_tier_thresholds(train_prices: pd.Series) -> Dict[str, float]:
    q = train_prices.quantile([0.25, 0.50, 0.75, 0.90])
    return {"q25": float(q.loc[0.25]), "q50": float(q.loc[0.50]), "q75": float(q.loc[0.75]), "q90": float(q.loc[0.90])}


def apply_price_tier(prices: pd.Series, thresholds: Dict[str, float]) -> pd.Categorical:
    bins = [-np.inf, thresholds["q25"], thresholds["q50"], thresholds["q75"], thresholds["q90"], np.inf]
    labels = ["Entry", "Lower-mid", "Mid", "Upper", "Premium"]
    return pd.cut(prices, bins=bins, labels=labels, include_lowest=True)


def make_user_input(
    reference_year: int,
    brand: str,
    model: str,
    year: int,
    mileage_km: float,
    origin: str,
    body_type: str,
    transmission: str,
    fuel_type: str,
    engine_size_l: float,
    exterior_color: str,
    seats: float,
    drivetrain: str,
) -> pd.DataFrame:
    mileage_zero_flag = int(float(mileage_km) == 0)
    mileage_value = np.nan if mileage_zero_flag else float(mileage_km)
    row = {
        "brand": brand,
        "model": model,
        "car_age": max(0, int(reference_year) - int(year)),
        "mileage_km": mileage_value,
        "mileage_zero_flag": mileage_zero_flag,
        "origin": origin,
        "body_type": body_type,
        "transmission": transmission,
        "fuel_type": fuel_type,
        "engine_size_l": float(engine_size_l) if engine_size_l is not None else np.nan,
        "exterior_color": exterior_color,
        "seats": float(seats) if seats is not None else np.nan,
        "drivetrain": drivetrain,
    }
    return pd.DataFrame([row], columns=FEATURES)


def compatibility_subset(
    model_df: pd.DataFrame,
    brand: str,
    model: str,
    year: Optional[int] = None,
    filters: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Return observed rows compatible with a brand/model/year and optional selections."""
    df = model_df[(model_df["brand"].astype(str) == str(brand)) & (model_df["model"].astype(str) == str(model))].copy()
    if year is not None:
        df = df[pd.to_numeric(df["year"], errors="coerce").eq(float(year))]
    for col, value in (filters or {}).items():
        if col not in df.columns or value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df = df[pd.to_numeric(df[col], errors="coerce").eq(float(value))]
        else:
            df = df[df[col].astype(str).eq(str(value))]
    return df


def compatible_values(df: pd.DataFrame, col: str) -> List:
    """Unique non-missing values in a deterministic order for the web form."""
    if col not in df.columns:
        return []
    s = df[col].dropna()
    if s.empty:
        return []
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
        return [float(v) for v in vals]
    return sorted(s.astype(str).unique().tolist())


def market_reference(
    model_df: pd.DataFrame, brand: str, model: str, year: int, min_exact: int = 4
) -> Dict[str, float]:
    """Train-data-only market context for display and a conservative plausibility guardrail."""
    split_col = model_df["split"] if "split" in model_df.columns else pd.Series(index=model_df.index, dtype=object)
    train = model_df[split_col.eq("train")].copy()
    if train.empty:
        train = model_df.copy()
    exact = compatibility_subset(train, brand, model, year)
    scope = "model-year"
    ref = exact
    if len(ref) < min_exact:
        ref = compatibility_subset(train, brand, model, None)
        scope = "model"
    prices = pd.to_numeric(ref[TARGET], errors="coerce").dropna() if TARGET in ref else pd.Series(dtype=float)
    if prices.empty:
        return {"count": 0, "scope": scope}
    return {
        "count": int(len(prices)),
        "scope": scope,
        "median": float(prices.median()),
        "q10": float(prices.quantile(0.10)),
        "q25": float(prices.quantile(0.25)),
        "q75": float(prices.quantile(0.75)),
        "q90": float(prices.quantile(0.90)),
    }


def apply_market_guardrail(raw_prediction: float, reference: Dict[str, float]) -> Tuple[float, bool]:
    """Clip only clearly implausible predictions; leave ordinary predictions untouched.

    Bounds are intentionally loose (0.75*q10 to 1.25*q90). The UI labels when
    a guardrail was applied, so the raw model estimate remains auditable.
    """
    if not reference or int(reference.get("count", 0)) < 4:
        return float(raw_prediction), False
    lo = 0.75 * float(reference["q10"])
    hi = 1.25 * float(reference["q90"])
    guarded = float(np.clip(float(raw_prediction), lo, hi))
    return guarded, not np.isclose(guarded, float(raw_prediction))


def build_web_options(model_df: pd.DataFrame, reference_year: int) -> Dict:
    df = model_df.copy()
    brand_models = {}
    for brand, g in df.dropna(subset=["brand", "model"]).groupby("brand"):
        brand_models[str(brand)] = sorted(g["model"].astype(str).unique().tolist())

    def choices(col):
        return sorted(df[col].dropna().astype(str).unique().tolist())

    return {
        "reference_year": int(reference_year),
        "year_min": int(df["year"].dropna().min()),
        "year_max": int(df["year"].dropna().max()),
        "brand_models": brand_models,
        "origin": choices("origin"),
        "body_type": choices("body_type"),
        "transmission": choices("transmission"),
        "fuel_type": choices("fuel_type"),
        "exterior_color": choices("exterior_color"),
        "drivetrain": choices("drivetrain"),
        "seats": sorted([int(x) for x in df["seats"].dropna().unique().tolist() if 1 <= x <= 60]),
        "engine_size_min": float(df["engine_size_l"].dropna().quantile(0.01)),
        "engine_size_max": float(df["engine_size_l"].dropna().quantile(0.99)),
    }
