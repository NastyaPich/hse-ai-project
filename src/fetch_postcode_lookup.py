"""
Скачивает для всех почтовых индексов из train/test район (borough) и код LSOA через postcodes.io
и сохраняет data/external/postcode_lookup.csv: postcode, borough, ward, lsoa_code, msoa_code.

Запуск из корня репозитория (нужен интернет, ~600 запросов по 100 индексов, займёт несколько минут):
    python src/fetch_postcode_lookup.py
Если прервать, повторный запуск докачает только недостающие индексы.

Источник: postcodes.io, данные ONS Postcode Directory (https://postcodes.io, https://api.postcodes.io/docs).
Эндпоинт: POST https://api.postcodes.io/postcodes, тело {"postcodes": [...]}, до 100 индексов за запрос
(по документации https://api.postcodes.io/docs/api/bulk-postcode-lookup).
Скрипт написан по документации и проверен только на макете ответа; на реальном API не запускался.
Проверьте результат: доля пустых borough должна быть близка к 0.
"""
import argparse
import json
import os
import time
import urllib.request

import pandas as pd

URL = "https://api.postcodes.io/postcodes"


def parse_bulk(payload):
    rows = []
    for item in payload.get("result", []):
        r = item.get("result")
        pc = item.get("query")
        if not r:
            rows.append({"postcode": pc, "borough": None, "ward": None, "lsoa_code": None, "msoa_code": None})
            continue
        codes = r.get("codes", {}) or {}
        rows.append({"postcode": pc, "borough": r.get("admin_district"), "ward": r.get("admin_ward"),
                     "lsoa_code": codes.get("lsoa"), "msoa_code": codes.get("msoa")})
    return rows


def fetch_batch(batch, retries=3):
    body = json.dumps({"postcodes": batch}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return parse_bulk(json.load(r))
        except Exception as e:  # сеть: подождать и повторить
            print("повтор после ошибки:", e)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("не удалось получить данные от postcodes.io")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/london", help="папка с train.csv и test.csv")
    ap.add_argument("--out", default="data/external/postcode_lookup.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pcs = pd.concat([pd.read_csv(os.path.join(a.data_dir, f"{n}.csv"), usecols=["postcode"])
                     for n in ("train", "test")])["postcode"].dropna().unique().tolist()
    done = pd.read_csv(a.out) if os.path.exists(a.out) else pd.DataFrame(columns=["postcode"])
    seen = set(done["postcode"])  # множество строим один раз, иначе проверка очень медленная
    todo = [p for p in pcs if p not in seen]
    print(f"индексов всего {len(pcs)}, осталось скачать {len(todo)}")
    rows = done.to_dict("records")
    for i in range(0, len(todo), 100):
        rows += fetch_batch(todo[i:i + 100])
        if (i // 100) % 20 == 0:
            pd.DataFrame(rows).to_csv(a.out, index=False)  # промежуточное сохранение, можно прервать и продолжить
        time.sleep(0.2)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(len(out), "строк ->", a.out, "| пустых borough:", out["borough"].isna().mean())
