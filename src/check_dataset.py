"""
Проверка, что собранная база совпала с той, по которой считались EDA и модели (из корня репозитория):

    python src/check_dataset.py                         # база в data/processed
    python src/check_dataset.py --dir "путь/к/папке"     # если собирали с --out-dir

Почему не побайтная контрольная сумма файла: разные версии pandas и numpy и разные системы (Windows, Linux)
по-разному записывают последние знаки дробных чисел и переносы строк, и сумма файла меняется,
хотя значения те же. Поэтому сравниваем содержимое: для каждого столбца число пропусков и
  - числовой столбец: сумма значений, округлённая до 6 значащих цифр;
  - текстовый столбец: отпечаток (md5) всех значений.
Из этого получается один короткий отпечаток на таблицу. Проверено: Linux (pandas 3.0.5, numpy 2.5.3)
и Windows (pandas 3.1.1, numpy 2.5.5) дают одинаковые отпечатки.
"""
import argparse
import hashlib
import os

import pandas as pd

EXPECTED = {"train": "1d2e6e0d4549", "test": "9b4348e5c35f"}  # отпечатки базы чекпойнтов 2-3


def column_fp(s):
    if pd.api.types.is_numeric_dtype(s):
        return f"{int(s.isna().sum())}:{float(s.sum()):.6g}"
    values = "\n".join(sorted(s.dropna().astype(str)))
    return f"{int(s.isna().sum())}:" + hashlib.md5(values.encode()).hexdigest()[:8]


def table_fp(path):
    d = pd.read_csv(path, low_memory=False)
    lines = [f"{len(d)}x{d.shape[1]}"] + [f"{c}:{column_fp(d[c])}" for c in d.columns]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed", help="папка с train_final.csv и test_final.csv")
    a = ap.parse_args()
    ok = True
    for name, expected in EXPECTED.items():
        got = table_fp(os.path.join(a.dir, f"{name}_final.csv"))
        same = got == expected
        ok &= same
        print(f"{name}_final.csv: {got} (ожидается {expected}) - {'совпадает' if same else 'НЕ совпадает'}")
    print("База совпала с эталонной." if ok else "База отличается от эталонной: проверьте исходные и внешние данные.")


if __name__ == "__main__":
    main()
