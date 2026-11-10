"""
Собирает основную базу одной командой (из корня репозитория):

    python src/build_dataset.py

Шаги:
  1. читает data/raw/london/train.csv и test.csv;
  2. очистка (src/clean.py): строки не удаляются, добавляются флаги и pt_group;
     пропуски заполняются в новых столбцах *_filled (исходные столбцы не меняются);
  3. признаки (src/add_features.py): признаки из цен считаются только по "чистой" истории
     (train без дубликатов и подозрительных цен);
  4. сохраняет data/processed/{train,test}_enriched.csv - все признаки, включая отброшенные
     (нужны для повторной проверки вклада групп, src/ablation.py);
  5. сохраняет data/processed/{train,test}_final.csv - итоговый набор: только нужные столбцы
     (списки RAW, FLAGS, FILLED, KEEP, KEYS ниже; почему именно они - FEATURES.md).

Только шаг 5 (заново выбрать столбцы из готовых *_enriched.csv, без пересчёта признаков):
    python src/build_dataset.py --only-final

Внешние файлы по умолчанию берутся из data/external (как их получить - README.md, раздел «Данные»).
Если какого-то файла нет, соответствующая группа пропускается, а итоговые таблицы не пишутся.
Списки столбцов итогового набора можно импортировать из другого кода: import build_dataset as bd; bd.KEEP.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import add_features as af  # noqa: E402
import clean  # noqa: E402

# --- итоговый набор столбцов (решение по итогам абляции, см. FEATURES.md) ---
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
    """Оставляет в обогащённой таблице только столбцы итогового набора (в train ещё и price)."""
    cols = RAW + (["price"] if is_train else []) + [c for c in FLAGS + FILLED if c in df.columns] + KEEP + KEYS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Нет столбцов {missing}: соберите таблицы со всеми внешними файлами "
                         "(станции, почтовые индексы, IMD в data/external)")
    return df[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/london")
    ap.add_argument("--external-dir", default="data/external")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--only-final", action="store_true",
                    help="только шаг 5: выбрать итоговые столбцы из готовых *_enriched.csv в --out-dir")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.only_final:
        for name in ("train", "test"):
            d = pd.read_csv(os.path.join(a.out_dir, f"{name}_enriched.csv"))
            out = select_final(d, name == "train")
            out.to_csv(os.path.join(a.out_dir, f"{name}_final.csv"), index=False)
            print(f"{name}: {out.shape[0]} строк, {out.shape[1]} столбцов -> {name}_final.csv")
        return

    train = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(a.data_dir, "test.csv"))
    print(f"train {train.shape}, test {test.shape}")

    def ext(name):
        p = os.path.join(a.external_dir, name)
        if not os.path.exists(p):
            print(f"Нет файла {p}: группа признаков будет пропущена")
            return None
        return p

    st_p, pc_p, imd_p = ext("stations_london.csv"), ext("postcode_lookup.csv"), ext("ID 2019 for London.xlsx")
    stations = af.load_stations(st_p) if st_p else None
    postcodes = af.load_postcode_lookup(pc_p) if pc_p else None
    imd = af.load_imd(imd_p) if (imd_p and postcodes is not None) else None

    # 2. очистка
    tr, te = clean.add_clean_flags(train, test)
    tr, te = clean.fill_missing(tr, te)
    print("\nФлаги очистки:\n" + clean.summary(tr, te).to_string())
    hist = tr[clean.history_mask(tr)]
    print(f"История для признаков из цен: {len(hist)} из {len(tr)} строк train")

    # 3. признаки
    tr, te = af.build(tr, te, stations, postcodes, imd, hist=hist[train.columns])
    for name, orig, d in (("train", train, tr), ("test", test, te)):
        assert len(d) == len(orig), "число строк изменилось"
        assert list(orig.columns) == list(d.columns[:len(orig.columns)]), "исходные столбцы изменились"

    # 4-5. сохранение
    for name, d in (("train", tr), ("test", te)):
        p = os.path.join(a.out_dir, f"{name}_enriched.csv")
        d.to_csv(p, index=False)
        print(f"{name}: {d.shape[0]} строк, {d.shape[1]} столбцов -> {p}")
    if stations is None or postcodes is None or imd is None:
        print("Итоговые таблицы не записаны: нужны все внешние файлы.")
        return
    for name, d in (("train", tr), ("test", te)):
        out = select_final(d, name == "train")
        p = os.path.join(a.out_dir, f"{name}_final.csv")
        out.to_csv(p, index=False)
        print(f"{name}: {out.shape[0]} строк, {out.shape[1]} столбцов -> {p}")
        new = [c for c in KEEP if c != "borough"]
        print((out[new].isna().mean() * 100).round(1).rename("пропусков, %").to_string())


if __name__ == "__main__":
    main()
