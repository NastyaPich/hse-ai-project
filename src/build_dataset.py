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
     (нужны для повторной проверки вклада групп, src/ablation_multi.py);
  5. сохраняет data/processed/{train,test}_final.csv - итоговый набор (src/make_final.py).

Внешние файлы по умолчанию берутся из data/external (см. data/README.md).
Если какого-то файла нет, соответствующая группа пропускается, а итоговые таблицы не пишутся.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import add_features as af  # noqa: E402
import clean  # noqa: E402
import make_final as mf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/london")
    ap.add_argument("--external-dir", default="data/external")
    ap.add_argument("--out-dir", default="data/processed")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

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
        out = mf.select_final(d, name == "train")
        p = os.path.join(a.out_dir, f"{name}_final.csv")
        out.to_csv(p, index=False)
        print(f"{name}: {out.shape[0]} строк, {out.shape[1]} столбцов -> {p}")
        new = [c for c in mf.KEEP if c != "borough"]
        print((out[new].isna().mean() * 100).round(1).rename("пропусков, %").to_string())


if __name__ == "__main__":
    main()
