"""
Абляция на нескольких годах: для каждого года Y из YEARS обучаемся на продажах до Y-1, проверяем на Y.
Признаки с ценами (группы 3, 4, 6) пересчитываются заново только по истории до Y-1 (как для настоящего test).
Статические признаки (расстояния, транспорт, район, IMD, ставка) не зависят от цен и берутся из обогащённых таблиц.

    python src/ablation_multi.py --train data/processed/train_enriched.csv
Нужна таблица со всеми группами (add_features.py с --stations, --postcodes, --imd).
Результат: reports/ablation_multi_result.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import add_features as af
import ablation as ab

RAW = ["ID", "fullAddress", "postcode", "country", "outcode", "latitude", "longitude", "bathrooms", "bedrooms",
       "floorAreaSqM", "livingRooms", "tenure", "propertyType", "currentEnergyRating", "sale_month", "sale_year", "price"]
STATIC = ab.OPTIONAL["транспорт"] + ["borough", "lsoa_code"] + ab.OPTIONAL["индекс депривации IMD"]
YEARS = [2021, 2022, 2023]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train_enriched.csv или pkl")
    ap.add_argument("--out", default="reports/ablation_multi_result.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    enr = pd.read_pickle(a.train) if a.train.endswith(".pkl") else pd.read_csv(a.train)
    raw = enr[RAW]
    static = enr[["ID"] + STATIC].set_index("ID")
    groups = dict(ab.GROUPS); groups.update(ab.OPTIONAL)
    rows = []
    for y in YEARS:
        part, val = raw[raw.sale_year < y], raw[raw.sale_year == y]
        tr, va = af.build(part, val.drop(columns="price"))
        tr = tr.join(static, on="ID"); va = va.join(static, on="ID")
        va = va.assign(price=val["price"].to_numpy())
        sets = [("базовые столбцы", ab.BASE)]
        sets += [(f"+ {g}", ab.BASE + c) for g, c in groups.items()]
        sets.append(("+ все группы", ab.BASE + sum(groups.values(), [])))
        sets += [(f"все, но без: {g}", ab.BASE + sum((c for k, c in groups.items() if k != g), [])) for g in groups]
        for name, feats in sets:
            rmse, r2 = ab.fit_eval(tr, va, feats)
            rows.append({"год проверки": y, "набор": name, "RMSE": rmse, "R2": r2})
            print(y, name, round(rmse, 4), round(r2, 4), flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
