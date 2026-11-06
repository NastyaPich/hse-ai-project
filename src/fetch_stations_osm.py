"""
Скачивает станции метро и железной дороги Большого Лондона из OpenStreetMap (Overpass API)
и сохраняет data/external/stations_london.csv: name, lat, lon, kind, open_year.

Запуск из корня репозитория (нужен интернет):  python src/fetch_stations_osm.py

Источник: OpenStreetMap contributors, лицензия ODbL (https://www.openstreetmap.org/copyright).
Ограничения: open_year берётся из тега start_date и заполнен лишь у части станций;
пустое значение означает "год открытия неизвестен". Проверьте результат глазами:
в Большом Лондоне должно быть порядка 600-700 станций всех видов.
"""
import argparse
import json
import os
import re
import urllib.parse
import urllib.request

import pandas as pd

BBOX = "51.28,-0.52,51.70,0.34"  # юг, запад, север, восток - Большой Лондон с запасом
QUERY = f"""
[out:json][timeout:180];
(
  nwr["railway"="station"]({BBOX});
  nwr["railway"="halt"]({BBOX});
);
out tags center;
"""
URL = "https://overpass-api.de/api/interpreter"


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/external/stations_london.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    payload = download(QUERY)
    st = parse(payload)
    st.to_csv(a.out, index=False)
    print(len(st), "станций сохранено в", a.out)
    print(st["kind"].value_counts().to_string())
