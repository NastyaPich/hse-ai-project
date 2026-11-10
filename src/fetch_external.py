"""
Скачивание внешних данных для признаков (нужен интернет). Запуск из корня репозитория:

    python src/fetch_external.py stations    # станции -> data/external/stations_london.csv
    python src/fetch_external.py postcodes   # район и LSOA -> data/external/postcode_lookup.csv

Третий внешний файл, IMD 2019 ("ID 2019 for London.xlsx"), скачивается вручную с London Datastore
(https://data.london.gov.uk/dataset/indices-of-deprivation) в data/external/.
Файлы из data/external в git не хранятся. Как они превращаются в признаки - src/add_features.py и FEATURES.md.

1) stations: станции метро и железной дороги Большого Лондона из OpenStreetMap (Overpass API).
   Столбцы: name, lat, lon, kind, open_year.
   Источник: OpenStreetMap contributors, лицензия ODbL (https://www.openstreetmap.org/copyright).
   Ограничения: open_year берётся из тега start_date и заполнен лишь у части станций;
   пустое значение означает "год открытия неизвестен". Проверьте результат глазами:
   в Большом Лондоне должно быть порядка 600-700 станций всех видов.

2) postcodes: для всех почтовых индексов из train/test район (borough) и код LSOA через postcodes.io.
   Столбцы: postcode, borough, ward, lsoa_code, msoa_code.
   ~600 запросов по 100 индексов, займёт несколько минут. Если прервать, повторный запуск
   докачает только недостающие индексы.
   Источник: postcodes.io, данные ONS Postcode Directory (https://postcodes.io, https://api.postcodes.io/docs).
   Эндпоинт: POST https://api.postcodes.io/postcodes, тело {"postcodes": [...]}, до 100 индексов за запрос
   (по документации https://api.postcodes.io/docs/api/bulk-postcode-lookup).
   Скрипт написан по документации и проверен только на макете ответа; на реальном API не запускался.
   Проверьте результат: доля пустых borough должна быть близка к 0.
"""
import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

# ---------------------------------------------------------------------------------------------------
# 1. Станции (OpenStreetMap, Overpass API)
# ---------------------------------------------------------------------------------------------------
BBOX = "51.28,-0.52,51.70,0.34"  # юг, запад, север, восток - Большой Лондон с запасом
QUERY = f"""
[out:json][timeout:180];
(
  nwr["railway"="station"]({BBOX});
  nwr["railway"="halt"]({BBOX});
);
out tags center;
"""
def classify(tags):
    network = (tags.get("network") or "") + ";" + (tags.get("operator") or "")
    if tags.get("station") == "subway" or "London Underground" in network:
        return "tube"
    if "light_rail" in tags.get("station", "") or "Docklands" in network:
        return "other"
    return "rail"


def parse(payload):
    rows = []
    for el in payload.get("elements", []):
        tags = el.get("tags", {})
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or "name" not in tags:
            continue
        m = re.match(r"(\d{4})", tags.get("start_date", ""))
        rows.append({"name": tags["name"], "lat": lat, "lon": lon, "kind": classify(tags),
                     "open_year": int(m.group(1)) if m else None})
    df = pd.DataFrame(rows)
    # одна станция в OSM часто описана несколькими объектами - склеиваем по имени и округлённым координатам
    df["_k"] = df["name"] + df["lat"].round(3).astype(str) + df["lon"].round(3).astype(str)
    return df.drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)


MIRRORS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter",
           "https://lz4.overpass-api.de/api/interpreter"]


def download(query):
    """Overpass отклоняет запросы без User-Agent (ошибка 406), поэтому заголовки заданы явно;
    при сбое пробуем запасные серверы."""
    data = urllib.parse.urlencode({"data": query}).encode()
    headers = {"User-Agent": "hse-ai-project-london-eda/1.0 (student project)",
               "Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    last = None
    for url in MIRRORS:
        try:
            print("Запрашиваю", url)
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=240) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            print("  не получилось:", e)
    raise SystemExit(f"Ни один сервер Overpass не ответил: {last}")

def fetch_stations(out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    payload = download(QUERY)
    st = parse(payload)
    st.to_csv(out, index=False)
    print(len(st), "станций сохранено в", out)
    print(st["kind"].value_counts().to_string())


# ---------------------------------------------------------------------------------------------------
# 2. Район и LSOA по почтовому индексу (postcodes.io)
# ---------------------------------------------------------------------------------------------------
POSTCODES_URL = "https://api.postcodes.io/postcodes"


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
            req = urllib.request.Request(POSTCODES_URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return parse_bulk(json.load(r))
        except Exception as e:  # сеть: подождать и повторить
            print("повтор после ошибки:", e)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("не удалось получить данные от postcodes.io")

def fetch_postcodes(data_dir, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pcs = pd.concat([pd.read_csv(os.path.join(data_dir, f"{n}.csv"), usecols=["postcode"])
                     for n in ("train", "test")])["postcode"].dropna().unique().tolist()
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["postcode"])
    seen = set(done["postcode"])  # множество строим один раз, иначе проверка очень медленная
    todo = [p for p in pcs if p not in seen]
    print(f"индексов всего {len(pcs)}, осталось скачать {len(todo)}")
    rows = done.to_dict("records")
    for i in range(0, len(todo), 100):
        rows += fetch_batch(todo[i:i + 100])
        if (i // 100) % 20 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)  # промежуточное сохранение, можно прервать и продолжить
        time.sleep(0.2)
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    print(len(res), "строк ->", out, "| пустых borough:", res["borough"].isna().mean())


def main():
    ap = argparse.ArgumentParser(description="Скачивание внешних данных (нужен интернет).")
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("stations", help="станции из OpenStreetMap -> stations_london.csv")
    s.add_argument("--out", default="data/external/stations_london.csv")
    p = sub.add_parser("postcodes", help="район и LSOA с postcodes.io -> postcode_lookup.csv")
    p.add_argument("--data-dir", default="data/raw/london", help="папка с train.csv и test.csv")
    p.add_argument("--out", default="data/external/postcode_lookup.csv")
    a = ap.parse_args()
    if a.what == "stations":
        fetch_stations(a.out)
    else:
        fetch_postcodes(a.data_dir, a.out)


if __name__ == "__main__":
    main()
