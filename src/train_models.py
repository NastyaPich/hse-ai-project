"""
Чекпойнт 3: первые ML-модели. Все расчёты для notebooks/ML.ipynb (ноутбук только читает таблицы из reports/).

Запуск из корня репозитория (сначала база: python src/build_dataset.py):
    python src/train_models.py                    # все шаги по порядку (~30 минут на 4 ядрах)
    python src/train_models.py --stage knn        # один шаг

Шаги (--stage):
  folds      временные фолды 2021-2023 и «фолд» train -> test 2024 (кэш data/processed/folds/), их описание
  baselines  ориентиры без ML: медиана цены м² района × площадь; прошлая цена адреса
  knn        KNN: окно лет, набор признаков, k и веса, масштаб координат
  linear     LinearRegression, Ridge (признаки по шагам, подбор alpha), Lasso
  lgb        LightGBM: по умолчанию, подбор гиперпараметров, работа с данными и признаками
  analysis   ошибки по сегментам, важность признаков, гипотеза про площадь
  final      лучшая модель на всём train -> прогноз для test 2024

Проверка только по времени: фолд Y = обучение на годах < Y, проверка на годе Y; признаки из цен пересчитаны
по истории до Y-1 (src/ml_utils.py). Подбор гиперпараметров - по 2021-2022, 2023 - честная проверка подбора.
Решения о признаках и данных: изменение оставляем, если оно лучше в среднем по 2021-2023 и не меньше чем
в 2 годах из 3. Цель - log(цена за м²); ключевая метрика - RMSE по log(цены) (в цене).

Результаты (reports/):
  ml_folds.csv         фолды: сколько строк, сколько исключено
  ml_results.csv       все модели и опыты по годам: ключевая метрика и вспомогательные
  ml_tuning.csv        подбор гиперпараметров (KNN, Ridge, Lasso, LightGBM): настройки -> ошибка по годам
  ml_errors.csv        ошибки по сегментам (KNN, Ridge, LightGBM)
  ml_importance.csv    важность признаков: SHAP для LightGBM, коэффициенты Ridge
  ml_area.csv          гипотеза «больше площадь - ниже цена за м²»
  test_predictions.csv прогноз для test 2024 (ID, predicted_price)
  ml_best.json         выбранные настройки (нужны следующим шагам)
Модель: models/lgbm_final.txt + models/lgbm_final_meta.json (признаки, настройки, категории).
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_utils as mu  # noqa: E402

R = "reports"
warnings.filterwarnings("ignore")  # предупреждения о сходимости Lasso и устаревших параметрах не нужны в выводе
BEST = os.path.join(R, "ml_best.json")


# --- сохранение результатов ------------------------------------------------------------------------
def save(df, name):
    p = os.path.join(R, name)
    df.to_csv(p, index=False)
    print(f"-> {p}", flush=True)


def put(path, new, key):
    """Дописывает строки в csv; старые строки с теми же значениями key заменяются."""
    p = os.path.join(R, path)
    if os.path.exists(p):
        old = pd.read_csv(p)
        old = old[~old[key].isin(new[key].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(p, index=False)


def add_result(res, section, model, final=False):
    """Строки модели/опыта в ml_results.csv. final=True - модель попадает в итоговую таблицу."""
    res = res.assign(раздел=section, модель=model, итог=final)
    cols = ["раздел", "модель", "итог", "год", "проверка"] + mu.METRICS + ["n"]
    put("ml_results.csv", res[cols], "модель")
    s = mu.per_year(res)
    print(f"{model}: RMSE_log {np.mean(list(s.values())):.4f} | " + ", ".join(f"{y}: {v:.4f}" for y, v in s.items()),
          flush=True)


def add_tuning(rows, model):
    put("ml_tuning.csv", pd.DataFrame(rows).assign(модель=model), "модель")


def load_best():
    if os.path.exists(BEST):
        with open(BEST) as f:
            return json.load(f)
    return {}


def save_best(**kw):
    b = load_best()
    b.update(kw)
    with open(BEST, "w") as f:
        json.dump(b, f, ensure_ascii=False, indent=1)


def mean_rmse(res, years=mu.YEARS):
    s = mu.per_year(res)
    return float(np.mean([s[y] for y in years]))


def helps(new, ref):
    """Изменение оставляем, если оно лучше в среднем по 2021-2023 и не меньше чем в 2 годах из 3."""
    a, b = mu.per_year(new), mu.per_year(ref)
    wins = sum(a[y] < b[y] for y in mu.YEARS)
    return mean_rmse(new) < mean_rmse(ref) and wins >= 2


def tuning_row(res, stage, settings, **params):
    s = mu.per_year(res)
    return {"этап": stage, "настройки": settings, **params, **{f"RMSE_{y}": s.get(y, np.nan) for y in mu.YEARS},
            "RMSE_подбор": float(np.mean([s[y] for y in mu.TUNE_YEARS]))}


# --- шаг folds ------------------------------------------------------------------------------------
def stage_folds(folds):
    rows = []
    for y, (tr, va) in folds.items():
        rows.append({"год проверки": y, "обучение: годы": f"1995–{y - 1}", "обучение: строк": len(tr),
                     "обучение: дубликаты": int(tr["is_duplicate"].sum()),
                     "обучение: подозрительная цена (price_suspect)": int(tr["price_suspect"].sum()),
                     "обучение: ошибочная цена без площади (price_suspect_noarea)": int(tr["price_suspect_noarea"].sum()),
                     "обучение: используем строк": int(mu.train_mask(tr).sum()),
                     "проверка: строк": len(va), "проверка: дубликаты": int(va["is_duplicate"].sum()),
                     "проверка: используем строк": int(mu.val_mask(va).sum()),
                     "проверка: из них подозрительная цена": int((~mu.clean_rows(va[mu.val_mask(va)])).sum()),
                     "проверка: есть прошлая продажа, %": float(va["has_prev"].mean() * 100),
                     "проверка: площадь пропущена, %": float(va["floorAreaSqM_missing"].mean() * 100)})
    save(pd.DataFrame(rows), "ml_folds.csv")


# --- шаг baselines: ориентиры без ML --------------------------------------------------------------
def outcode_median_price(tr, va):
    """Медиана цены за м² в outcode за прошлый год (по «чистым» продажам с известной площадью) × площадь.
    Если в outcode за прошлый год продаж нет - медиана по всему Лондону за прошлый год."""
    y = int(va["sale_year"].iloc[0]) - 1
    h = tr[mu.train_mask(tr) & (tr["sale_year"] == y).to_numpy() & tr["floorAreaSqM"].notna().to_numpy()]
    ppsm = h["price"] / h["floorAreaSqM"]
    med = ppsm.groupby(h["outcode"]).median()
    return va["outcode"].map(med).fillna(ppsm.median()).to_numpy() * mu.area(va)


def prev_price(tr, va):
    """Прошлая цена того же адреса с поправкой на рост цен района; если прошлой продажи нет - ориентир 0а."""
    p = va["prev_price_adj"].to_numpy()
    return np.where(np.isnan(p), outcode_median_price(tr, va), p)


def stage_baselines(folds):
    add_result(mu.cv(folds, outcode_median_price), "ориентир", "0а. Медиана цены м² района × площадь", True)
    add_result(mu.cv(folds, prev_price), "ориентир", "0б. Прошлая цена адреса (иначе 0а)", True)


# --- шаг knn ---------------------------------------------------------------------------------------
KNN_START = {"feature_set": "место + объект", "k": 5, "weights": "uniform", "coord_weight": 1.0, "window": 30}
KNN_STAGES = [  # (этап, параметр, варианты); каждый этап берёт лучшее по 2021-2022 и идёт дальше
    ("1. окно лет обучения", "window", [1, 2, 3, 5, 8, 12, 30]),
    ("2. набор признаков", "feature_set", list(mu.KNN_SETS)),
    ("3. число соседей k и веса", ("k", "weights"), [(k, w) for w in ("uniform", "distance")
                                                      for k in (5, 10, 20, 30, 50, 100)]),
    ("4. вес координат", "coord_weight", [0.5, 1.0, 2.0, 3.0]),
]
MIN_GAIN = 0.001
KNN_NAMES = ["KNN: начальный (k = 5, место + объект, все годы)", "KNN: + окно лет", "KNN: + набор признаков",
             "KNN: + подбор k и весов", "KNN: + вес координат (итог)"]


def run_knn(folds, p):
    return mu.cv(folds, lambda tr, va: mu.fit_knn(tr, va, mu.KNN_SETS[p["feature_set"]], p["k"], p["weights"],
                                                  p["coord_weight"], p["window"]))


def knn_settings(p):
    return (f"признаки: {p['feature_set']}; k = {p['k']}; веса: {p['weights']}; вес координат {p['coord_weight']}; "
            f"окно {p['window']} лет")


def stage_knn(folds):
    cur, rows, cache = dict(KNN_START), [], {}

    def run(p):
        key = json.dumps(p, sort_keys=True)
        if key not in cache:
            t = time.time()
            cache[key] = run_knn(folds, p)
            print(f"  {knn_settings(p)}: {mean_rmse(cache[key], mu.TUNE_YEARS):.4f} ({time.time() - t:.0f} c)", flush=True)
        return cache[key]

    add_result(run(cur), "KNN", KNN_NAMES[0], True)
    for (stage, param, values), name in zip(KNN_STAGES, KNN_NAMES[1:]):
        # новое значение берём, только если оно лучше текущего хотя бы на MIN_GAIN (иначе это шум)
        cur_score, cand = mean_rmse(run(cur), mu.TUNE_YEARS), None
        for v in values:
            p = dict(cur)
            if isinstance(param, tuple):
                p.update(dict(zip(param, v)))
            else:
                p[param] = v
            r = run(p)
            rows.append(tuning_row(r, stage, knn_settings(p), **p))
            if cand is None or mean_rmse(r, mu.TUNE_YEARS) < cand[0]:
                cand = (mean_rmse(r, mu.TUNE_YEARS), p)
        if cand[0] < cur_score - MIN_GAIN:
            cur = cand[1]
        rows[-len(values):] = [dict(x, выбрано=(x["настройки"] == knn_settings(cur))) for x in rows[-len(values):]]
        add_result(run(cur), "KNN", name, name == KNN_NAMES[-1])
    add_tuning(rows, "KNN")
    save_best(knn=cur)
    print("KNN выбрано:", knn_settings(cur))


# --- шаг linear ------------------------------------------------------------------------------------
LIN_START = ["floorAreaSqM_filled", "bedrooms_filled", "bathrooms_filled", "livingRooms_filled",
             "latitude", "longitude", "sale_month", "sale_year"]
LIN_KEEP = ["has_prev", "prev_rel_knn", "prev_sale_gap_months", "dist_charing_cross_km", "dist_canary_wharf_km",
            "dist_nearest_centre_km", "dist_subcentre_km", "dist_station_km", "dist_tube_km", "n_addr_500m",
            "knn20_prev_med_logppsm", "imd_score", "crime_decile", "living_env_decile"]
MISSING_FLAGS = ["floorAreaSqM_missing", "bedrooms_missing", "bathrooms_missing", "livingRooms_missing"]


def lin_steps():
    """Шаги feature engineering для линейной модели: (название, функция cfg -> новый cfg)."""
    def rep(lst, a, b):
        return [b if c == a else c for c in lst]
    return [
        ("площадь -> log площади", lambda c: {**c, "num": rep(c["num"], "floorAreaSqM_filled", "log_area")}),
        ("+ признаки чекпойнта 2 (прошлая продажа, расстояния, транспорт, соседи, IMD, borough)",
         lambda c: {**c, "num": c["num"] + LIN_KEEP, "cat": c["cat"] + ["borough"]}),
        ("без года продажи (уровень цен - через цены соседей прошлого года)",
         lambda c: {**c, "num": [x for x in c["num"] if x != "sale_year"]}),
        ("+ флаги пропусков", lambda c: {**c, "flags": c["flags"] + MISSING_FLAGS}),
        ("+ квадраты и взаимодействия", lambda c: {**c, "num": c["num"] + mu.POLY}),
        ("+ флаг «адрес на несколько квартир» (по прошлому)",
         lambda c: {**c, "flags": c["flags"] + ["addr_multi_unit_past"]}),
        ("обучение только на последних 10 годах", lambda c: {**c, "window": 10}),
    ]


def run_linear(folds, cfg, model="ridge", alpha=1.0, years=None):
    return mu.cv(folds, lambda tr, va: mu.fit_linear(tr, va, cfg["num"], cfg["flags"], cfg["cat"], model, alpha,
                                                     cfg.get("window")), years)


def stage_linear(folds):
    cfg = {"num": LIN_START, "flags": [], "cat": list(mu.LIN_CAT), "window": None}
    add_result(run_linear(folds, cfg, "linreg"), "линейные", "LinearRegression: исходные столбцы", True)
    ref = run_linear(folds, cfg)
    add_result(ref, "линейные", "Ridge (alpha = 1): исходные столбцы")
    # feature engineering по шагам: шаг остаётся, только если помогает (helps)
    for i, (name, f) in enumerate(lin_steps(), 1):
        new_cfg = f(cfg)
        r = run_linear(folds, new_cfg)
        ok = helps(r, ref)
        add_result(r, "линейные", f"Ridge, шаг {i}: {name}" + ("" if ok else " (не взяли)"))
        if ok:
            cfg, ref = new_cfg, r
    # подбор alpha на итоговом наборе (по 2021-2022)
    rows = []
    for a in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        r = run_linear(folds, cfg, "ridge", a)
        rows.append(tuning_row(r, "alpha", f"alpha = {a}", alpha=a))
        print(f"  Ridge alpha {a}: {rows[-1]['RMSE_подбор']:.4f}", flush=True)
    # при почти равной ошибке (разница < 0,0001) берём большее alpha - более «осторожную» модель
    top = min(x["RMSE_подбор"] for x in rows)
    best_a = max(x["alpha"] for x in rows if x["RMSE_подбор"] < top + 1e-4)
    for x in rows:
        x["выбрано"] = x["alpha"] == best_a
    add_tuning(rows, "Ridge")
    add_result(run_linear(folds, cfg, "ridge", best_a), "линейные",
               f"Ridge: признаки + подбор alpha (итог, alpha = {best_a:g})", True)
    add_result(run_linear(folds, cfg, "linreg"), "линейные", "LinearRegression: итоговые признаки")
    # Lasso: сам обнуляет лишние коэффициенты
    rows, best = [], None
    for a in [1e-5, 1e-4, 1e-3]:
        r = run_linear(folds, cfg, "lasso", a)
        rows.append(tuning_row(r, "alpha", f"alpha = {a:g}", alpha=a))
        print(f"  Lasso alpha {a}: {rows[-1]['RMSE_подбор']:.4f}", flush=True)
        if best is None or rows[-1]["RMSE_подбор"] < best[0]:
            best = (rows[-1]["RMSE_подбор"], a, r)
    for x in rows:
        x["выбрано"] = x["alpha"] == best[1]
    # сколько коэффициентов Lasso обнулил (на фолде 2023)
    tr, va = folds[2023]
    _, pipe = mu.fit_linear(tr, va, cfg["num"], cfg["flags"], cfg["cat"], "lasso", best[1], cfg.get("window"),
                            return_model=True)
    coef = pipe.named_steps["model"].coef_
    rows[[x["alpha"] for x in rows].index(best[1])]["ненулевых коэффициентов (2023)"] = f"{int((coef != 0).sum())} из {len(coef)}"
    add_tuning(rows, "Lasso")
    add_result(best[2], "линейные", f"Lasso: итоговые признаки (alpha = {best[1]:g})", True)
    save_best(linear={**cfg, "alpha": best_a})


# --- шаг lgb ---------------------------------------------------------------------------------------
LGB_BASE = mu.BASE_RAW + mu.KEEP  # исходные столбцы (с пропусками) + 15 признаков чекпойнта 2
SEARCH_N = 25
SPACE = {
    "n_estimators": [300, 500, 800, 1200],
    "learning_rate": [0.03, 0.05, 0.08],
    "num_leaves": [31, 63, 127, 255],
    "min_child_samples": [10, 20, 50, 100],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "reg_lambda": [0.0, 1.0, 5.0],
}
# работа с данными и признаками: (название, изменение признаков, параметры обучения)
LGB_EXPERIMENTS = [
    ("строки: обучение на всех строках (с дубликатами и подозрительными ценами)", None, {"drop_flagged": False}),
    ("строки: без правила price_suspect_noarea (исключаем только price_suspect)", None, {"use_noarea": False}),
    ("пропуски: заполненные *_filled вместо пропусков", "filled", {}),
    ("+ цена м² прошлой продажи (log_prev_ppsm_adj, prev_rel_knn)", ["log_prev_ppsm_adj", "prev_rel_knn"], {}),
    ("+ флаг «адрес на несколько квартир» (по прошлому)", ["addr_multi_unit_past"], {}),
    ("+ флаги пропусков и подозрительной площади", MISSING_FLAGS + ["area_suspect"], {}),
]


def run_lgb(folds, feats, params, years=None, **kw):
    return mu.cv(folds, lambda tr, va: mu.fit_lgb(tr, va, feats, params, **kw), years)


def lgb_settings(p):
    return ", ".join(f"{k} {v}" for k, v in p.items() if k != "subsample_freq")


def stage_lgb(folds):
    default = run_lgb(folds, LGB_BASE, {})
    add_result(default, "LightGBM", "LightGBM: по умолчанию", True)
    # 1. случайный поиск по 2021-2022; 2023 - только для лучшей комбинации
    rng = np.random.default_rng(42)
    rows = [tuning_row(default, "по умолчанию", "по умолчанию (100 деревьев, learning_rate 0.1, num_leaves 31)")]
    for i in range(SEARCH_N):
        p = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
        p = {k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}
        p["subsample_freq"] = 1 if p["subsample"] < 1 else 0
        t = time.time()
        r = run_lgb(folds, LGB_BASE, p, mu.TUNE_YEARS)
        rows.append(tuning_row(r, "случайный поиск", lgb_settings(p), **p, секунд=round(time.time() - t)))
        print(f"  {i}: {lgb_settings(p)} -> {rows[-1]['RMSE_подбор']:.4f} ({rows[-1]['секунд']} c)", flush=True)
    best = min(rows[1:], key=lambda x: x["RMSE_подбор"])
    params = {k: best[k] for k in list(SPACE) + ["subsample_freq"]}
    tuned = run_lgb(folds, LGB_BASE, params)
    best.update({f"RMSE_{y}": v for y, v in mu.per_year(tuned).items()})
    for x in rows:
        x["выбрано"] = x is best
    add_tuning(rows, "LightGBM")
    add_result(tuned, "LightGBM", "LightGBM: подбор гиперпараметров", True)
    # 2. работа с данными и признаками: каждое изменение отдельно относительно модели после подбора
    feats, accepted = list(LGB_BASE), []
    for name, change, kw in LGB_EXPERIMENTS:
        if change == "filled":
            f = mu.BASE_FILLED + mu.KEEP
        else:
            f = LGB_BASE + (change or [])
        r = run_lgb(folds, f, params, **kw)
        ok = helps(r, tuned)
        add_result(r, "LightGBM", f"LightGBM, опыт: {name}")
        if ok and not kw:  # правила отбора строк приняты заранее; признаки - по правилу helps
            accepted.append(name)
            feats = (mu.BASE_FILLED + [c for c in feats if c not in mu.BASE_RAW]) if change == "filled" \
                else feats + change
    final = run_lgb(folds, feats, params) if accepted else tuned
    add_result(final, "LightGBM", "LightGBM: подбор + признаки (итог)", True)
    add_result(run_lgb(folds, feats, params, target="price"), "LightGBM",
               "справка: итоговый LightGBM с целью log(цены)")
    save_best(lgb={"features": feats, "params": params, "accepted": accepted})
    print("LightGBM: приняты изменения", accepted)


# --- шаг analysis ----------------------------------------------------------------------------------
def price_group(p):
    bins = [0, 300_000, 500_000, 750_000, 1_000_000, 2_000_000, np.inf]
    labels = ["< 300 тыс.", "300–500 тыс.", "500–750 тыс.", "750 тыс.–1 млн", "1–2 млн", "> 2 млн"]
    return pd.cut(p, bins=bins, labels=labels, right=False).astype(str)


def stage_analysis(folds):
    b = load_best()
    lin, knn, lg = b["linear"], b["knn"], b["lgb"]
    models = {
        "KNN": lambda tr, va: mu.fit_knn(tr, va, mu.KNN_SETS[knn["feature_set"]], knn["k"], knn["weights"],
                                         knn["coord_weight"], knn["window"]),
        "Ridge": lambda tr, va: mu.fit_linear(tr, va, lin["num"], lin["flags"], lin["cat"], "ridge", lin["alpha"],
                                              lin.get("window")),
        "LightGBM": lambda tr, va: mu.fit_lgb(tr, va, lg["features"], lg["params"]),
    }
    oof = []
    for y, (tr, va) in folds.items():
        va = va[mu.val_mask(va)]
        d = va[["ID", "price", "pt_group_filled", "has_prev", "addr_multi_unit", "floorAreaSqM_missing", "borough",
                "sale_year"]].copy()
        d["clean"] = mu.clean_rows(va)
        for name, f in models.items():
            d[f"pred_{name}"] = f(tr, va)
        oof.append(d)
        print("прогнозы фолда", y, flush=True)
    oof = pd.concat(oof, ignore_index=True)
    seg = {
        "тип жилья": oof["pt_group_filled"],
        "прошлая продажа": oof["has_prev"].map({1: "есть", 0: "нет"}),
        "площадь была пропущена": oof["floorAreaSqM_missing"].map({1: "да", 0: "нет"}),
        "адрес на несколько квартир": oof["addr_multi_unit"].map({1: "да", 0: "нет"}),
        "подозрительная цена": oof["clean"].map({True: "нет", False: "да"}),
        "цена (реальная)": price_group(oof["price"]),
        "borough": oof["borough"].fillna("нет данных"),
    }
    rows = []
    for name in models:
        pred = oof[f"pred_{name}"]
        log_err = np.log(pred / oof["price"])
        seg_m = dict(seg, **{"цена (прогноз)": price_group(pred)})
        for s, g in seg_m.items():
            for val, idx in oof.groupby(g.to_numpy()).groups.items():
                m = mu.metrics(oof.loc[idx, "price"], pred.loc[idx])
                rows.append({"модель": name, "сегмент": s, "значение": val, **m,
                             "смещение_%": float(np.median(log_err.loc[idx]) * 100)})
        # крупнейшие промахи: какая доля из них - подозрительные цены
        top = log_err.abs().nlargest(100).index
        for val, idx in oof.loc[top].groupby(oof.loc[top, "clean"].map({True: "нет", False: "да"})).groups.items():
            rows.append({"модель": name, "сегмент": "100 крупнейших промахов: подозрительная цена", "значение": val,
                         **mu.metrics(oof.loc[idx, "price"], pred.loc[idx]),
                         "смещение_%": float(np.median(log_err.loc[idx]) * 100)})
    save(pd.DataFrame(rows), "ml_errors.csv")
    importance(folds, lin, lg)
    area_regression()


def importance(folds, lin, lg):
    """SHAP для LightGBM и коэффициенты Ridge на фолде 2023 (обучение до 2022 года)."""
    import shap
    tr, va = folds[2023]
    va = va[mu.val_mask(va)]
    _, model, X_va = mu.fit_lgb(tr, va, lg["features"], lg["params"], return_model=True)
    Xs = X_va.sample(3000, random_state=0)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    imp = pd.DataFrame({"источник": "LightGBM: SHAP", "признак": lg["features"],
                        "значение": np.abs(sv).mean(axis=0)}).sort_values("значение", ascending=False)
    # для чтения коэффициентов берём Ridge без квадратов и взаимодействий: у них коэффициенты зависят друг от друга
    # (например, log_area и log_area_sq), и по одному их не прочитать
    num = [c for c in lin["num"] if c not in mu.POLY]
    _, pipe = mu.fit_linear(tr, va, num, lin["flags"], lin["cat"], "ridge", lin["alpha"], lin.get("window"),
                            return_model=True)
    pre, reg = pipe.named_steps["pre"], pipe.named_steps["model"]
    names = pre.get_feature_names_out()
    co = pd.DataFrame({"источник": "Ridge (без квадратов): коэффициент", "признак": names, "значение": reg.coef_})
    co["группа"] = co["признак"].str.split("__").str[0]
    co["признак"] = co["признак"].str.split("__", n=1).str[1]
    # числа: эффект изменения на 1 стандартное отклонение (стд - в исходных единицах);
    # флаги и категории: «да» против «нет»/базовой группы. эффект_% - изменение цены за м² в %
    co["стд"] = np.nan
    co.loc[co["группа"] == "num", "стд"] = pre.named_transformers_["num"][-1].scale_
    co["эффект_%"] = (np.exp(co["значение"]) - 1) * 100
    save(pd.concat([imp, co], ignore_index=True), "ml_importance.csv")


def area_regression():
    """Гипотеза «больше площадь - ниже цена за м²»: линейная регрессия log(цена за м²) на log(площадь)
    с учётом года, района (outcode) и типа жилья. Это объясняющая модель на всём «чистом» train, не прогноз.
    Коэффициент b: при росте площади на 1% цена за м² меняется примерно на b%."""
    from scipy import sparse
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder
    d = mu.add_derived(pd.read_csv(mu.ENRICHED["train"], low_memory=False))
    d = d[(d["is_duplicate"] == 0) & (d["price_suspect"] == 0) & d["floorAreaSqM"].notna() & (d["area_suspect"] == 0)]
    y = np.log(d["price"] / d["floorAreaSqM"]).to_numpy()
    la = np.log(d["floorAreaSqM"].to_numpy())
    specs = [("1. только площадь", []), ("2. + год продажи", ["sale_year"]),
             ("3. + год и район (outcode)", ["sale_year", "outcode"]),
             ("4. + год, район, тип жилья", ["sale_year", "outcode", "pt_group_filled"]),
             ("5. + год, район, тип, спальни, энергокласс, tenure",
              ["sale_year", "outcode", "pt_group_filled", "bedrooms_filled", "currentEnergyRating_filled",
               "tenure_filled"])]
    rows = []
    for period, mask in (("1995–2023", np.ones(len(d), bool)), ("2019–2023", (d["sale_year"] >= 2019).to_numpy())):
        for name, cats in specs:
            X = sparse.csr_matrix(la[mask].reshape(-1, 1))
            if cats:
                X = sparse.hstack([X, OneHotEncoder(drop="first").fit_transform(d.loc[mask, cats].astype(str))]).tocsr()
            b = float(LinearRegression().fit(X, y[mask]).coef_[0])
            rows.append({"период": period, "модель": name, "коэффициент при log площади": b,
                         "изменение цены за м² при площади ×2, %": (2 ** b - 1) * 100, "n": int(mask.sum())})
    save(pd.DataFrame(rows), "ml_area.csv")


# --- шаг final -------------------------------------------------------------------------------------
def stage_final(folds):
    lg = load_best()["lgb"]
    tr, te = folds["test"]
    price, model, _ = mu.fit_lgb(tr, te, lg["features"], lg["params"], return_model=True)
    os.makedirs("models", exist_ok=True)
    model.booster_.save_model("models/lgbm_final.txt")
    t = tr[mu.train_mask(tr)]
    with open("models/lgbm_final_meta.json", "w") as f:
        json.dump({"target": "log(price / floorAreaSqM_filled)", "features": lg["features"], "params": lg["params"],
                   "train_rows": len(t),
                   "categories": {c: sorted(t[c].dropna().astype(str).unique().tolist())
                                  for c in lg["features"] if c in mu.CATS}}, f, ensure_ascii=False, indent=1)
    out = pd.DataFrame({"ID": te["ID"], "predicted_price": np.round(price).astype(int)})
    save(out, "test_predictions.csv")
    print(out["predicted_price"].describe().round(0).to_string())


STAGES = ["folds", "baselines", "knn", "linear", "lgb", "analysis", "final"]


def main():
    ap = argparse.ArgumentParser(description="Модели чекпойнта 3: расчёты по шагам, результаты в reports/.")
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES,
                    help="шаг расчёта (по умолчанию все по порядку)")
    a = ap.parse_args()
    os.makedirs(R, exist_ok=True)
    stages = STAGES if a.stage == "all" else [a.stage]
    allf = mu.build_folds(final=True)
    folds = {y: allf[y] for y in mu.YEARS}
    for s in stages:
        t = time.time()
        print(f"\n=== {s} ===", flush=True)
        globals()[f"stage_{s}"](allf if s == "final" else folds)
        print(f"=== {s}: {time.time() - t:.0f} c ===", flush=True)


if __name__ == "__main__":
    main()
