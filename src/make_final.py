"""
Оставляет в обогащённых таблицах только нужные столбцы (решение по итогам проверки, см. FEATURES.md).

Обычно вызывается из src/build_dataset.py. Отдельный запуск:
    python src/make_final.py
Читает data/processed/train_enriched.csv и test_enriched.csv,
пишет data/processed/train_final.csv и test_final.csv.
"""
import argparse
import os

import pandas as pd

RAW = ["ID", "fullAddress", "postcode", "outcode", "latitude", "longitude", "bathrooms", "bedrooms",
       "floorAreaSqM", "livingRooms", "tenure", "propertyType", "currentEnergyRating", "sale_month", "sale_year"]
# столбец country (всегда England) убран как неинформативный
# флаги очистки (src/clean.py): строки не удаляются, решение о них принимается при обучении
FLAGS = ["pt_group", "is_duplicate", "dup_addr_month", "price_suspect", "area_suspect", "addr_multi_unit"]
# заполненные пропуски (src/clean.py, fill_missing): исходные столбцы остаются как есть
FILLED = ["floorAreaSqM_filled", "bedrooms_filled", "bathrooms_filled", "livingRooms_filled",
          "floorAreaSqM_missing", "bedrooms_missing", "bathrooms_missing", "livingRooms_missing",
          "tenure_filled", "currentEnergyRating_filled", "pt_group_filled"]
KEEP = [
    # прошлая продажа того же адреса (только из прошлого)
    "prev_sale_price", "prev_price_adj", "prev_sale_gap_months",
    # расстояния до центров Лондона
    "dist_charing_cross_km", "dist_canary_wharf_km", "dist_nearest_centre_km", "dist_subcentre_km",
    # транспорт
    "dist_station_km", "dist_tube_km",
    # окружение и район
    "n_addr_500m", "knn20_prev_med_logppsm", "borough",
    # индекс депривации IMD 2019
    "imd_score", "crime_decile", "living_env_decile",
]
KEYS = ["lsoa_code"]  # справочный столбец для присоединения новых данных (в модель не идёт)


def select_final(df, is_train):
    cols = RAW + (["price"] if is_train else []) + [c for c in FLAGS + FILLED if c in df.columns] + KEEP + KEYS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Нет столбцов {missing}: соберите таблицы со всеми внешними файлами "
                         "(--stations, --postcodes, --imd)")
    return df[cols]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/processed")
    ap.add_argument("--out-dir", default="data/processed")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for name in ("train", "test"):
        d = pd.read_csv(os.path.join(a.in_dir, f"{name}_enriched.csv"))
        out = select_final(d, name == "train")
        out.to_csv(os.path.join(a.out_dir, f"{name}_final.csv"), index=False)
        print(f"{name}: {out.shape[0]} строк, {out.shape[1]} столбцов -> {name}_final.csv")
