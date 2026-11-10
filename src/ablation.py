"""
Абляция: помогает ли каждая группа признаков предсказывать log(цена)?

Идея: обучаем одну и ту же модель (HistGradientBoosting) с группой признаков и без неё и сравниваем ошибку.
Проверка только по времени: для каждого года Y обучаемся на продажах до Y-1 и проверяем на Y (по умолчанию
Y = 2021, 2022, 2023). Признаки с ценами (группы «цены района», «прошлая продажа», «цены ближайших продаж»)
для каждого года пересчитываются заново только по истории до Y-1 (как для настоящего test).
Признаки без цен (транспорт, район, IMD) не зависят от года и берутся из готовой таблицы *_enriched.csv.
Если в таблице есть флаги очистки (build_dataset.py), признаки из цен считаются только по строкам без
дубликатов и подозрительных цен, как в основной базе. Обучение и проверка идут на всех строках.

Запуск из корня репозитория (около 30 минут на три года):
    python src/ablation.py --train data/processed/train_enriched.csv
Быстрая проверка на одном году:
    python src/ablation.py --train data/processed/train_enriched.csv --years 2023 --out /tmp/abl.csv
Нужна таблица со всеми группами (python src/build_dataset.py с файлами из data/external).
Результат: reports/ablation_multi_result.csv (его читает notebooks/EDA.ipynb, раздел 7.9).
Выводы и таблица результатов: FEATURES.md.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import add_features as af  # noqa: E402
import build_dataset as bd  # noqa: E402
import clean  # noqa: E402

# исходные столбцы, с которыми сравниваем все группы
BASE = ["bathrooms", "bedrooms", "floorAreaSqM", "livingRooms", "tenure", "propertyType",
        "currentEnergyRating", "outcode", "latitude", "longitude", "sale_month", "sale_year"]
# группы признаков (src/add_features.py); порядок групп важен: от него зависит порядок столбцов модели
GROUPS = {
    "расстояния до центров": ["dist_charing_cross_km", "dist_city_km", "dist_canary_wharf_km",
                              "dist_nearest_centre_km", "dist_subcentre_km", "dist_west_end_km"],
    "ставка Банка Англии": ["bank_rate_pct", "bank_rate_chg_12m"],
    "цены района за прошлый год": ["oc_prev_med_logprice", "oc_prev_med_logppsm", "oc_prev_n",
                                   "oc_price_yoy", "sec_prev_med_logppsm", "sec_prev_n"],
    "прошлая продажа адреса": ["prev_sale_price", "prev_sale_gap_months", "n_prev_sales", "prev_price_adj"],
    "плотность и цены ближайших продаж": ["n_addr_500m", "knn20_prev_med_logppsm", "knn20_prev_dist_km"],
}
# группы из внешних файлов (станции, почтовые индексы, IMD), см. src/fetch_external.py
OPTIONAL = {
    "транспорт": ["dist_station_km", "dist_tube_km", "n_stations_1km"],
    "район (borough)": ["borough"],
    "индекс депривации IMD": ["imd_score", "imd_decile", "crime_decile", "living_env_decile", "income_decile"],
}
CATS = ["tenure", "propertyType", "currentEnergyRating", "outcode", "borough"]

# исходные столбцы train.csv: из них признаки с ценами пересчитываются для каждого года
RAW = ["ID", "fullAddress", "postcode", "country", "outcode", "latitude", "longitude", "bathrooms", "bedrooms",
       "floorAreaSqM", "livingRooms", "tenure", "propertyType", "currentEnergyRating", "sale_month", "sale_year", "price"]
# признаки без цен: берутся из готовой таблицы как есть
STATIC = OPTIONAL["транспорт"] + ["borough", "lsoa_code"] + OPTIONAL["индекс депривации IMD"]
YEARS = [2021, 2022, 2023]
# название строки итогового набора в reports/ablation_multi_result.csv
# (список KEEP раньше лежал в make_final.py, теперь в build_dataset.py; название оставлено, чтобы файл не менялся)
FINAL_NAME = "итоговый набор (make_final.KEEP)"


def fit_eval(tr, va, feats):
    """Обучает HistGradientBoosting на tr по столбцам feats и возвращает (RMSE по log цены, R²) на va."""
    X_tr, X_va = tr[feats].copy(), va[feats].copy()
    cats = [c for c in CATS if c in feats]
    if cats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
        X_tr[cats] = enc.fit_transform(X_tr[cats].astype("object"))
        X_va[cats] = enc.transform(X_va[cats].astype("object"))
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=0)
    m.fit(X_tr, np.log(tr["price"]))
    p = m.predict(X_va)
    y = np.log(va["price"].to_numpy())
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return rmse, float(1 - np.mean((y - p) ** 2) / np.var(y))


def feature_sets():
    """Список (название, столбцы): база, база + одна группа, все группы, все без одной группы, итоговый набор."""
    groups = dict(GROUPS)
    groups.update(OPTIONAL)
    sets = [("базовые столбцы", BASE)]
    sets += [(f"+ {g}", BASE + c) for g, c in groups.items()]
    sets.append(("+ все группы", BASE + sum(groups.values(), [])))
    sets += [(f"все, но без: {g}", BASE + sum((c for k, c in groups.items() if k != g), [])) for g in groups]
    sets.append((FINAL_NAME, BASE + bd.KEEP))
    return sets


def year_split(enr, y):
    """Обучающая (годы < y) и проверочная (год y) части с признаками, посчитанными только по истории до y-1."""
    raw = enr[RAW]
    ok = clean.history_mask(enr) if "price_suspect" in enr.columns else pd.Series(True, index=enr.index)
    static = enr[["ID"] + STATIC].set_index("ID")
    part, val = raw[raw.sale_year < y], raw[raw.sale_year == y]
    tr, va = af.build(part, val.drop(columns="price"), hist=part[ok.loc[part.index]])
    tr = tr.join(static, on="ID")
    va = va.join(static, on="ID")
    return tr, va.assign(price=val["price"].to_numpy())


def main():
    ap = argparse.ArgumentParser(description="Абляция групп признаков с проверкой по времени.")
    ap.add_argument("--train", required=True, help="train_enriched.csv (или .pkl)")
    ap.add_argument("--out", default="reports/ablation_multi_result.csv")
    ap.add_argument("--years", type=int, nargs="+", default=YEARS, help="годы проверки (по умолчанию 2021 2022 2023)")
    ap.add_argument("--sets", nargs="+", help="посчитать только эти наборы (по названию; для быстрой проверки)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    enr = pd.read_pickle(a.train) if a.train.endswith(".pkl") else pd.read_csv(a.train)
    sets = [s for s in feature_sets() if not a.sets or s[0] in a.sets]
    rows = []
    for y in a.years:
        tr, va = year_split(enr, y)
        for name, feats in sets:
            rmse, r2 = fit_eval(tr, va, feats)
            rows.append({"год проверки": y, "набор": name, "RMSE": rmse, "R2": r2})
            print(y, name, round(rmse, 4), round(r2, 4), flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
