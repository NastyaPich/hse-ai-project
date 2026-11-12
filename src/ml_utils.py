"""
Общие функции для моделей чекпойнта 3 (их вызывает src/train_models.py).

Что здесь есть:
  load_base        - готовая база (data/processed/*_enriched.csv) + правило очистки чекпойнта 3
                     price_suspect_noarea (src/clean.py). Сама база при этом не меняется.
  build_folds      - временные фолды 2021, 2022, 2023: обучение на годах < Y, проверка на Y.
                     Признаки из цен (цены района, прошлая продажа, цены соседей) для каждого фолда
                     пересчитываются ТОЛЬКО по истории до Y-1 (как в src/ablation.py); пропуски заполняются
                     по медианам train-части фолда. Тем же способом строится «фолд» весь train -> test 2024.
  add_derived      - производные признаки (log площади, цена за м² прошлой продажи и т.п.).
  metrics          - ключевая метрика RMSE по log(цены) и вспомогательные (MdAPE, доли в ±10% и ±20%, MAE).
  knn_pipeline, linear_pipeline, fit_lgb - три вида моделей.

Цель всех моделей: log(цена за м²), площадь = floorAreaSqM_filled (исходная, если известна, иначе заполненная).
Прогноз цены = exp(прогноз) * площадь. Все метрики считаются в цене.

Защита от утечек: медианы для заполнения пропусков, масштаб (StandardScaler), кодирование категорий
(OneHotEncoder, список категорий LightGBM) обучаются только на train-части фолда; test в обучении не участвует.
"""
import os
import sys
from functools import partial

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import add_features as af  # noqa: E402
import build_dataset as bd  # noqa: E402
import clean  # noqa: E402

YEARS = [2021, 2022, 2023]
TUNE_YEARS = [2021, 2022]  # подбор гиперпараметров - по этим годам; 2023 - честная проверка подбора
ENRICHED = {"train": "data/processed/train_enriched.csv", "test": "data/processed/test_enriched.csv"}
FOLD_DIR = "data/processed/folds"  # кэш фолдов (в git не попадает: data/processed/ в .gitignore)

RAW = ["ID", "fullAddress", "postcode", "outcode", "latitude", "longitude", "bathrooms", "bedrooms",
       "floorAreaSqM", "livingRooms", "tenure", "propertyType", "currentEnergyRating", "sale_month",
       "sale_year", "price"]
# флаги очистки: только для отбора строк и разбора ошибок (addr_multi_unit смотрит и на будущие даты,
# поэтому в модели не идёт)
FLAGS = ["pt_group", "is_duplicate", "dup_addr_month", "price_suspect", "area_suspect", "addr_multi_unit",
         "price_suspect_noarea"]
# признаки без цен (станции, район, IMD) не зависят от года - берём из готовой базы
STATIC = ["dist_station_km", "dist_tube_km", "borough", "lsoa_code", "imd_score", "crime_decile",
          "living_env_decile"]
SUBSET_CLEAN = "без подозрительных цен"

# --- наборы признаков ------------------------------------------------------------------------------
# базовые столбцы: с пропусками (NaN) и с заполненными значениями (*_filled)
BASE_RAW = ["bathrooms", "bedrooms", "floorAreaSqM", "livingRooms", "tenure", "pt_group",
            "currentEnergyRating", "outcode", "latitude", "longitude", "sale_month", "sale_year"]
BASE_FILLED = ["bathrooms_filled", "bedrooms_filled", "floorAreaSqM_filled", "livingRooms_filled",
               "tenure_filled", "pt_group_filled", "currentEnergyRating_filled", "outcode",
               "latitude", "longitude", "sale_month", "sale_year"]
KEEP = list(bd.KEEP)  # 15 признаков, отобранных на чекпойнте 2 (FEATURES.md)
CATS = ["tenure", "pt_group", "currentEnergyRating", "outcode", "borough", "tenure_filled",
        "pt_group_filled", "currentEnergyRating_filled"]


# --- загрузка и фолды ------------------------------------------------------------------------------
def load_base():
    """train и test из основной базы + флаг price_suspect_noarea (правило очистки чекпойнта 3).
    Флаг считается по всему train один раз: это правило очистки (сравнение цены с медианой похожих продаж
    того же outcode и года), а не признак модели."""
    tr = pd.read_csv(ENRICHED["train"], low_memory=False)
    te = pd.read_csv(ENRICHED["test"], low_memory=False)
    return clean.add_noarea_flag(tr, te)


def build_split(enr_tr, enr_va):
    """(train-часть, проверка) с признаками, посчитанными только по прошлому для проверки.
    enr_tr - продажи до года проверки, enr_va - продажи года проверки (или test без цены)."""
    has_price = "price" in enr_va.columns
    cols = [c for c in RAW + FLAGS if c in enr_tr.columns]
    part, val = enr_tr[cols], enr_va[[c for c in cols if c in enr_va.columns]]
    # пропуски: медианы только по train-части
    part, val_np = clean.fill_missing(part, val.drop(columns="price", errors="ignore"))
    # признаки из цен: только по «чистой» истории train-части (без дубликатов и подозрительных цен)
    hist = part[clean.history_mask_strict(part)]
    tr, va = af.build(part, val_np, hist=hist[RAW])
    # «адрес на несколько квартир» только по прошлым датам продаж train-части
    tr["addr_multi_unit_past"] = clean.addr_multi_unit_past(tr, part)
    va["addr_multi_unit_past"] = clean.addr_multi_unit_past(va, part)
    static = pd.concat([enr_tr, enr_va])[["ID"] + STATIC].drop_duplicates("ID").set_index("ID")
    tr, va = tr.join(static, on="ID"), va.join(static, on="ID")
    if has_price:
        va = va.assign(price=enr_va["price"].to_numpy())
    return add_derived(tr), add_derived(va)


def build_folds(years=YEARS, final=False, cache=True):
    """Словарь {год: (train, val)}; при final=True ещё ключ 'test': (весь train, test 2024).
    Результат кэшируется в data/processed/folds/*.pkl (удалить папку, если база пересобрана)."""
    keys = list(years) + (["test"] if final else [])
    out, base = {}, None
    for y in keys:
        p_tr, p_va = (os.path.join(FOLD_DIR, f"{y}_{k}.pkl") for k in ("train", "val"))
        if cache and os.path.exists(p_tr) and os.path.exists(p_va):
            out[y] = (pd.read_pickle(p_tr), pd.read_pickle(p_va))
            continue
        if base is None:
            base = load_base()
        tr_all, te_all = base
        if y == "test":
            tr, va = build_split(tr_all, te_all)
        else:
            tr, va = build_split(tr_all[tr_all.sale_year < y], tr_all[tr_all.sale_year == y])
        if cache:
            os.makedirs(FOLD_DIR, exist_ok=True)
            tr.to_pickle(p_tr)
            va.to_pickle(p_va)
        out[y] = (tr, va)
        print(f"фолд {y}: обучение {len(tr)} строк, проверка {len(va)} строк", flush=True)
    return out


def add_derived(df):
    """Производные признаки из уже посчитанных столбцов (без новых данных):
    log_area          - log площади;
    log_prev_ppsm_adj - log(цена прошлой продажи с поправкой на рост района / площадь);
    prev_rel_knn      - насколько эта цена за м² выше/ниже уровня 20 ближайших продаж прошлого года;
    has_prev          - есть ли прошлая продажа (0/1);
    is_flat, is_freehold - тип жилья «квартира» и форма собственности freehold (0/1)."""
    d = df.copy()
    a = d["floorAreaSqM_filled"].astype(float)
    d["log_area"] = np.log(a)
    d["log_prev_ppsm_adj"] = np.log(d["prev_price_adj"] / a)
    d["prev_rel_knn"] = d["log_prev_ppsm_adj"] - d["knn20_prev_med_logppsm"]
    d["has_prev"] = d["prev_sale_price"].notna().astype(int)
    d["is_flat"] = (d["pt_group_filled"] == "Flat").astype(int)
    d["is_freehold"] = (d["tenure_filled"] == "Freehold").astype(int)
    return d


def train_mask(tr, drop_flagged=True, use_noarea=True):
    """Строки для обучения: без полных дубликатов и подозрительных цен (price_suspect и price_suspect_noarea)."""
    if not drop_flagged:
        return np.ones(len(tr), bool)
    m = (tr["is_duplicate"] == 0) & (tr["price_suspect"] == 0)
    if use_noarea:
        m &= tr["price_suspect_noarea"] == 0
    return m.to_numpy()


def val_mask(va):
    """Строки для проверки: все, кроме полных дубликатов."""
    return (va["is_duplicate"] == 0).to_numpy()


def clean_rows(df):
    """Строки без подозрительных цен (оба правила)."""
    return ((df["price_suspect"] == 0) & (df["price_suspect_noarea"] == 0)).to_numpy()


# --- цель и метрики --------------------------------------------------------------------------------
def area(df):
    return df["floorAreaSqM_filled"].to_numpy(dtype=float)


def target_ppsm(df):
    """Цель: log(цена за м²)."""
    return np.log(df["price"].to_numpy(dtype=float) / area(df))


def to_price(df, pred_log_ppsm):
    """Прогноз log(цены за м²) -> прогноз цены, £."""
    return np.exp(pred_log_ppsm) * area(df)


METRICS = ["RMSE_log", "MdAPE_%", "в ±10%", "в ±20%", "MAE_£"]


def metrics(price_true, price_pred):
    """Ключевая метрика RMSE_log и вспомогательные. Всё в цене (£)."""
    y = np.asarray(price_true, dtype=float)
    p = np.asarray(price_pred, dtype=float)
    ape = np.abs(p - y) / y
    return {
        "RMSE_log": float(np.sqrt(np.mean((np.log(p) - np.log(y)) ** 2))),  # ключевая
        "MdAPE_%": float(np.median(ape) * 100),   # типичная ошибка в %
        "в ±10%": float(np.mean(ape <= 0.10)),
        "в ±20%": float(np.mean(ape <= 0.20)),
        "MAE_£": float(np.mean(np.abs(p - y))),
        "n": int(len(y)),
    }


def eval_rows(va, price_pred, year, **info):
    """Две строки результата: все строки года (кроме дубликатов) и без подозрительных цен."""
    rows = []
    for subset, m in (("все", np.ones(len(va), bool)), (SUBSET_CLEAN, clean_rows(va))):
        rows.append({**info, "год": year, "проверка": subset, **metrics(va["price"].to_numpy()[m], price_pred[m])})
    return rows


def cv(folds, fit_predict, years=None, **info):
    """Обучает модель на каждом фолде и возвращает таблицу метрик.
    fit_predict(tr, va) -> прогноз ЦЕНЫ для va. tr - строки обучения (уже отобраны), va - строки проверки."""
    rows = []
    for y, (tr, va) in folds.items():
        if years and y not in years:
            continue
        va = va[val_mask(va)]
        rows += eval_rows(va, fit_predict(tr, va), y, **info)
    return pd.DataFrame(rows)


def per_year(res, metric="RMSE_log"):
    """{год: метрика} по строкам «все»."""
    r = res[res["проверка"] == "все"]
    return r.set_index("год")[metric].to_dict()


# --- KNN -------------------------------------------------------------------------------------------
# KNN ищет k самых похожих продаж и усредняет их log(цену за м²). «Похожесть» = расстояние между признаками,
# поэтому все признаки приводятся к одному масштабу (StandardScaler), иначе признак в больших единицах
# (например, срок с прошлой продажи в месяцах) заглушит остальные. One-hot для района не используем: десятки
# столбцов 0/1 сделали бы расстояние почти бессмысленным; район KNN видит через координаты.
KNN_SETS = {
    "место": ["latitude", "longitude"],
    "место + объект": ["latitude", "longitude", "log_area", "bedrooms_filled", "bathrooms_filled",
                       "is_flat", "is_freehold"],
    "место + объект + окружение": ["latitude", "longitude", "log_area", "bedrooms_filled", "bathrooms_filled",
                                   "is_flat", "is_freehold", "knn20_prev_med_logppsm", "dist_charing_cross_km",
                                   "dist_tube_km", "imd_score"],
    "место + объект + окружение + прошлая продажа": [
        "latitude", "longitude", "log_area", "bedrooms_filled", "bathrooms_filled", "is_flat", "is_freehold",
        "knn20_prev_med_logppsm", "dist_charing_cross_km", "dist_tube_km", "imd_score", "has_prev", "prev_rel_knn"],
}
COORDS = ["latitude", "longitude"]


def knn_frame(df, feats):
    X = df[feats].astype(float).copy()
    if "prev_rel_knn" in X:
        X["prev_rel_knn"] = X["prev_rel_knn"].fillna(0)  # нет прошлой продажи -> «как у соседей»
    return X


def knn_pipeline(feats, k=20, weights="uniform", coord_weight=1.0):
    """Заполнение пропусков медианой, масштаб, затем KNN. coord_weight > 1 делает координаты важнее
    остальных признаков (расстояние по карте весит больше)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    other = [c for c in feats if c not in COORDS]
    parts = [("coords", make_pipeline(StandardScaler(),
                                      FunctionTransformer(partial(np.multiply, coord_weight))), COORDS)]
    if other:
        parts.append(("other", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), other))
    return Pipeline([("pre", ColumnTransformer(parts)),
                     ("knn", KNeighborsRegressor(n_neighbors=k, weights=weights, n_jobs=4))])


def fit_knn(tr, va, feats, k=20, weights="uniform", coord_weight=1.0, window=3, max_rows=None, seed=0):
    """KNN по продажам последних window лет перед годом проверки (старые цены для соседей не годятся:
    уровень цен с 1995 года вырос в разы). max_rows - случайная подвыборка обучения (для проверки устойчивости)."""
    t = tr[train_mask(tr)]
    t = t[t["sale_year"] > t["sale_year"].max() - window]
    if max_rows and len(t) > max_rows:
        t = t.sample(max_rows, random_state=seed)
    pipe = knn_pipeline(feats, k, weights, coord_weight).fit(knn_frame(t, feats), target_ppsm(t))
    return to_price(va, pipe.predict(knn_frame(va, feats)))


# --- линейные модели -------------------------------------------------------------------------------
LIN_CAT = ["pt_group_filled", "tenure_filled", "currentEnergyRating_filled", "outcode"]


def linear_frame(df, num, flags, cat):
    """Признаки для линейной модели. У строк без прошлой продажи prev_rel_knn = 0 и срок = 0, has_prev = 0
    (коэффициент читается как «насколько прошлая продажа выше/ниже уровня соседей»)."""
    X = df[num + flags + cat].copy()
    for c in ("prev_rel_knn", "prev_sale_gap_months"):
        if c in X:
            X[c] = X[c].fillna(0)
    for c in cat:
        X[c] = X[c].astype("object").where(X[c].notna(), "Unknown").astype(str)
    return X


def linear_pipeline(num, flags, cat, model="ridge", alpha=1.0):
    """sklearn Pipeline: числа - заполнение пропусков медианой + StandardScaler; флаги 0/1 - как есть;
    категории - OneHotEncoder (редкие значения, меньше 20 продаж, объединяются). Всё учится на train-части."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("flag", "passthrough", flags),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
    ])
    reg = {"linreg": LinearRegression(), "ridge": Ridge(alpha=alpha),
           "lasso": Lasso(alpha=alpha, max_iter=2000, tol=1e-3)}[model]
    return Pipeline([("pre", pre), ("model", reg)])


def add_poly(df):
    """Квадраты и взаимодействия для линейной модели (она сама умеет только прямые линии):
    log_area_sq     - квадрат log площади (связь площади с ценой за м² может быть изогнутой);
    log_area_x_flat - log площади × «квартира» (у квартир и домов разный наклон);
    dist_cc_sq      - квадрат расстояния до центра (цена падает от центра не по прямой);
    knn_x_flat      - уровень цен соседей × «квартира»."""
    la = df["log_area"]
    return df.assign(log_area_sq=la ** 2, log_area_x_flat=la * df["is_flat"],
                     dist_cc_sq=df["dist_charing_cross_km"] ** 2,
                     knn_x_flat=df["knn20_prev_med_logppsm"] * df["is_flat"])


POLY = ["log_area_sq", "log_area_x_flat", "dist_cc_sq", "knn_x_flat"]


def fit_linear(tr, va, num, flags=(), cat=LIN_CAT, model="ridge", alpha=1.0, window=None, return_model=False):
    """Линейная модель. window - учиться только на продажах последних window лет (None - на всех)."""
    num, flags, cat = list(num), list(flags), list(cat)
    t = tr[train_mask(tr)]
    if window:
        t = t[t["sale_year"] > t["sale_year"].max() - window]
    if any(c in POLY for c in num):
        t, va = add_poly(t), add_poly(va)
    pipe = linear_pipeline(num, flags, cat, model, alpha).fit(linear_frame(t, num, flags, cat), target_ppsm(t))
    p = to_price(va, pipe.predict(linear_frame(va, num, flags, cat)))
    return (p, pipe) if return_model else p


# --- LightGBM --------------------------------------------------------------------------------------
def lgb_matrix(tr, va, feats):
    """Категории - pandas category со списком значений только из train-части (новое значение -> пропуск)."""
    X_tr, X_va = tr[feats].copy(), va[feats].copy()
    for c in feats:
        if c in CATS:
            cats = pd.Index(X_tr[c].dropna().astype(str).unique()).sort_values()
            X_tr[c] = pd.Categorical(X_tr[c].astype("string"), categories=cats)
            X_va[c] = pd.Categorical(X_va[c].astype("string"), categories=cats)
    return X_tr, X_va


def fit_lgb(tr, va, feats, params=None, target="ppsm", drop_flagged=True, use_noarea=True, return_model=False):
    """Обучает LightGBM и возвращает прогноз ЦЕНЫ для va. target: 'ppsm' - log(цена за м²), 'price' - log(цены)."""
    import lightgbm as lgb
    p = {"verbose": -1, "random_state": 0, "n_jobs": 4, **(params or {})}
    t = tr[train_mask(tr, drop_flagged, use_noarea)]
    X_tr, X_va = lgb_matrix(t, va, feats)
    y = np.log(t["price"].to_numpy(dtype=float)) if target == "price" else target_ppsm(t)
    model = lgb.LGBMRegressor(**p).fit(X_tr, y)
    pred = model.predict(X_va)
    price = np.exp(pred) if target == "price" else to_price(va, pred)
    return (price, model, X_va) if return_model else price
