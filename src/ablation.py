"""
Абляция: помогает ли каждая группа признаков предсказывать log(цена)?
Обучение на продажах до 2022 включительно, проверка на 2023 (как test: будущее относительно обучения).
Признаки для проверочной части считаются честно - только по истории до 2022 (функция build).

    python src/ablation.py --data-dir data/raw/london

Первая версия проверки: признаки считаются по всему train, без очистки (src/clean.py).
Актуальная проверка - src/ablation_multi.py (три года, очищенная база). Этот файл оставлен,
потому что ablation_multi.py берёт из него списки признаков (BASE, GROUPS, OPTIONAL) и fit_eval.
    (необязательно: --stations, --postcodes, --imd, как в add_features.py)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import add_features as af

BASE = ["bathrooms", "bedrooms", "floorAreaSqM", "livingRooms", "tenure", "propertyType",
        "currentEnergyRating", "outcode", "latitude", "longitude", "sale_month", "sale_year"]
GROUPS = {
    "расстояния до центров": ["dist_charing_cross_km", "dist_city_km", "dist_canary_wharf_km",
                              "dist_nearest_centre_km", "dist_subcentre_km", "dist_west_end_km"],
    "ставка Банка Англии": ["bank_rate_pct", "bank_rate_chg_12m"],
    "цены района за прошлый год": ["oc_prev_med_logprice", "oc_prev_med_logppsm", "oc_prev_n",
                                   "oc_price_yoy", "sec_prev_med_logppsm", "sec_prev_n"],
    "прошлая продажа адреса": ["prev_sale_price", "prev_sale_gap_months", "n_prev_sales", "prev_price_adj"],
    "плотность и цены ближайших продаж": ["n_addr_500m", "knn20_prev_med_logppsm", "knn20_prev_dist_km"],
}
# группы, которые появляются только при наличии внешних файлов (--stations, --postcodes, --imd)
OPTIONAL = {
    "транспорт": ["dist_station_km", "dist_tube_km", "n_stations_1km"],
    "район (borough)": ["borough"],
    "индекс депривации IMD": ["imd_score", "imd_decile", "crime_decile", "living_env_decile", "income_decile"],
}
CATS = ["tenure", "propertyType", "currentEnergyRating", "outcode", "borough"]


def fit_eval(tr, va, feats):
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stations")
    ap.add_argument("--postcodes")
    ap.add_argument("--imd")
    a = ap.parse_args()
    full = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    part, val = full[full.sale_year <= 2022], full[full.sale_year == 2023]
    stations = af.load_stations(a.stations) if a.stations else None
    postcodes = af.load_postcode_lookup(a.postcodes) if a.postcodes else None
    imd = af.load_imd(a.imd) if (a.imd and postcodes is not None) else None
    tr, va = af.build(part, val.drop(columns="price"), stations, postcodes, imd)
    for g, cols in OPTIONAL.items():
        if all(c in tr.columns for c in cols):
            GROUPS[g] = cols
    va = va.assign(price=val["price"].to_numpy())
    print(f"обучение: {len(tr)} строк (до 2022), проверка: {len(va)} строк (2023)\n")
    res = [("базовые столбцы", *fit_eval(tr, va, BASE))]
    for g, cols in GROUPS.items():
        res.append((f"+ {g}", *fit_eval(tr, va, BASE + cols)))
    res.append(("+ все группы", *fit_eval(tr, va, BASE + sum(GROUPS.values(), []))))
    for g, cols in GROUPS.items():
        others = sum((c for k, c in GROUPS.items() if k != g), [])
        res.append((f"все группы, но без: {g}", *fit_eval(tr, va, BASE + others)))
    print(pd.DataFrame(res, columns=["набор признаков", "RMSE log", "R²"]).round(3).to_string(index=False))
