"""
Обогащение таблиц London House Price (train.csv, test.csv) признаками.

Отдельно этот файл не запускают: функции вызывает src/build_dataset.py (очистка -> признаки -> итоговый
набор), а также src/ablation.py и код моделей. Главная функция - build(train, test, ...).
Внешние файлы (станции, почтовые индексы) скачивает src/fetch_external.py, файл IMD скачивается вручную.

Правило проекта: каждая группа признаков = отдельная функция add_*(df, ...), которая НЕ меняет
исходные столбцы, а только дописывает новые. Тогда вклад группы легко измерить абляцией
(обучить модель с группой и без, см. src/ablation.py).

Группы признаков (подробности и результаты проверки - в FEATURES.md):
  1. add_centre_distance        расстояние до центров Лондона и до ближайшего центра по London Plan
  2. add_bank_rate              ключевая ставка Банка Англии на момент продажи
  3. add_neighbourhood_history  уровень цен в районе за ПРЕДЫДУЩИЙ календарный год
  4. add_repeat_sale            предыдущая продажа того же адреса
  5. add_transport              расстояние до станций (нужен stations_london.csv)
  6. add_local_context          плотность застройки и цены 20 ближайших продаж за прошлый год
  7. add_postcode_area          район (borough) и LSOA по почтовому индексу (нужен postcode_lookup.csv)
  8. add_imd                    индекс депривации IMD 2019 по LSOA (нужен файл London Datastore)

ИСТОЧНИКИ ДАННЫХ
  Группы 3, 4, 6: только файл train.csv (и test.csv для плотности застройки) соревнования Kaggle
      "London House Price Prediction: Advanced Techniques"
      https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques
      Внешних данных нет, всё считается из этих таблиц.
  Группа 1: координаты центров - общеизвестные; классификация центров -
      London Plan 2021, Annex 1 Town centre network
      https://www.london.gov.uk/programmes-strategies/planning/london-plan/the-london-plan-2021-online/annex-1-town-centre-network
  Группа 2: Bank of England, Bank Rate history
      https://www.bankofengland.co.uk/boeapps/database/Bank-Rate
  Группа 5: OpenStreetMap contributors (ODbL), https://www.openstreetmap.org/copyright
      через Overpass API https://overpass-api.de ; теги https://wiki.openstreetmap.org/wiki/Key:railway
      альтернатива/сверка: открытые данные TfL https://tfl.gov.uk/info-for/open-data-users/
  Группа 7: postcodes.io (данные ONS Postcode Directory), https://postcodes.io , документация https://api.postcodes.io/docs
  Группа 8: English Indices of Deprivation 2019 (MHCLG), лицензия OGL v3;
      файл "ID 2019 for London.xlsx" на London Datastore: https://data.london.gov.uk/dataset/indices-of-deprivation

ВАЖНО про утечки: признаки 3, 4 и 6 используют цены, поэтому для строки они считаются только
по продажам, которые произошли ДО неё. Для test (2024) историей служит весь train (до 2023).
"""
import sys

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Расстояние по большой окружности в км (векторизовано)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Группа 1. Расстояние до центров
# ---------------------------------------------------------------------------
# Все объекты датасета лежат в почтовых районах Лондона (E, EC, N, NW, SE, SW, W, WC; широта
# 51.39-51.67, долгота -0.35...0.14), поэтому "ближайший крупный город" для всех один - Лондон.
# Внутри него есть крупные районные центры, расстояние до них даёт отдельный признак.
CENTRES = {  # (широта, долгота), WGS84, точность порядка 100 м
    "charing_cross": (51.5074, -0.1278),  # официальная точка отсчёта расстояний от Лондона
    "city_bank": (51.5133, -0.0886),      # Сити (Bank)
    "canary_wharf": (51.5049, -0.0195),   # второй деловой центр
}
# Центры по иерархии London Plan 2021, Annex 1 (Town centre network):
# https://www.london.gov.uk/programmes-strategies/planning/london-plan/the-london-plan-2021-online/annex-1-town-centre-network
# Классификация (international / metropolitan / major) взята оттуда и сверена 15.10.2026.
# Координаты центров - общеизвестные, точность порядка 0,5 км (для расстояний в км этого достаточно).
# В список вошли центры, лежащие в области данных или рядом с ней.
SUBCENTRES = {  # имя: (широта, долгота, уровень)
    "West End": (51.5152, -0.1415, "international"), "Knightsbridge": (51.5015, -0.1607, "international"),
    "Canary Wharf": (51.5049, -0.0195, "metropolitan"), "Croydon": (51.3762, -0.0982, "metropolitan"),
    "Kingston": (51.4123, -0.3007, "metropolitan"), "Bromley": (51.4060, 0.0143, "metropolitan"),
    "Stratford": (51.5416, -0.0034, "metropolitan"), "Wood Green": (51.5975, -0.1097, "metropolitan"),
    "Ealing": (51.5150, -0.3017, "metropolitan"), "Shepherds Bush": (51.5046, -0.2187, "metropolitan"),
    "Ilford": (51.5590, 0.0742, "metropolitan"), "Harrow": (51.5790, -0.3370, "metropolitan"),
    "Sutton": (51.3600, -0.1940, "metropolitan"),
    "Wembley": (51.5529, -0.2963, "major"), "Kilburn": (51.5347, -0.1938, "major"),
    "Camden Town": (51.5392, -0.1426, "major"), "Dalston": (51.5461, -0.0755, "major"),
    "Hackney Central": (51.5470, -0.0556, "major"), "Fulham": (51.4804, -0.1950, "major"),
    "Hammersmith": (51.4927, -0.2236, "major"), "Chiswick": (51.4925, -0.2550, "major"),
    "Angel": (51.5322, -0.1058, "major"), "Nags Head": (51.5530, -0.1190, "major"),
    "Kensington High Street": (51.5010, -0.1930, "major"), "Brixton": (51.4613, -0.1145, "major"),
    "Streatham": (51.4293, -0.1313, "major"), "Catford": (51.4457, -0.0267, "major"),
    "Lewisham": (51.4650, -0.0138, "major"), "Wimbledon": (51.4214, -0.2064, "major"),
    "East Ham": (51.5390, 0.0517, "major"), "Richmond": (51.4613, -0.3037, "major"),
    "Canada Water": (51.4980, -0.0500, "major"), "Elephant and Castle": (51.4946, -0.1005, "major"),
    "Peckham": (51.4739, -0.0690, "major"), "Walthamstow": (51.5830, -0.0200, "major"),
    "Clapham Junction": (51.4642, -0.1704, "major"), "Putney": (51.4614, -0.2160, "major"),
    "Tooting": (51.4275, -0.1680, "major"), "Wandsworth": (51.4560, -0.1910, "major"),
    "Woolwich": (51.4903, 0.0693, "major"), "Eltham": (51.4500, 0.0520, "major"),
    "Queensway/Westbourne Grove": (51.5115, -0.1880, "major"),
    "King's Road (east)": (51.4924, -0.1565, "major"),
}


def add_centre_distance(df):
    out = df.copy()
    names = {"charing_cross": "dist_charing_cross_km", "city_bank": "dist_city_km",
             "canary_wharf": "dist_canary_wharf_km"}
    for key, col in names.items():
        out[col] = haversine_km(out["latitude"], out["longitude"], *CENTRES[key])
    out["dist_nearest_centre_km"] = out[list(names.values())].min(axis=1)
    # ближайший центр из иерархии London Plan (любого уровня) и расстояние до West End
    names = list(SUBCENTRES)
    d = np.column_stack([haversine_km(out["latitude"], out["longitude"], v[0], v[1]) for v in SUBCENTRES.values()])
    j = d.argmin(axis=1)
    out["dist_subcentre_km"] = d.min(axis=1)
    out["nearest_subcentre"] = np.array(names)[j]
    out["subcentre_tier"] = np.array([SUBCENTRES[n][2] for n in names])[j]
    out["dist_west_end_km"] = d[:, names.index("West End")]
    return out


# ---------------------------------------------------------------------------
# Группа 2. Ключевая ставка Банка Англии
# ---------------------------------------------------------------------------
# Источник: Bank of England, "Bank Rate history and data"
# https://www.bankofengland.co.uk/boeapps/database/Bank-Rate  (таблица изменений ставки; загружено 15.10.2026).
# (дата изменения, новая ставка, %). Значения вида 5,94 или 6,13 - так они указаны в таблице Банка.
BANK_RATE_CHANGES = [
    ("1994-12-07", 6.13), ("1995-02-02", 6.63), ("1995-12-13", 6.38), ("1996-01-18", 6.13),
    ("1996-03-08", 5.94), ("1996-06-06", 5.69), ("1996-10-30", 5.94), ("1997-05-06", 6.25),
    ("1997-06-06", 6.50), ("1997-07-10", 6.75), ("1997-08-07", 7.00), ("1997-11-06", 7.25),
    ("1998-06-04", 7.50), ("1998-10-08", 7.25), ("1998-11-05", 6.75), ("1998-12-10", 6.25),
    ("1999-01-07", 6.00), ("1999-02-04", 5.50), ("1999-04-08", 5.25), ("1999-06-10", 5.00),
    ("1999-09-08", 5.25), ("1999-11-04", 5.50), ("2000-01-13", 5.75), ("2000-02-10", 6.00),
    ("2001-02-08", 5.75), ("2001-04-05", 5.50), ("2001-05-10", 5.25), ("2001-08-02", 5.00),
    ("2001-09-18", 4.75), ("2001-10-04", 4.50), ("2001-11-08", 4.00), ("2003-02-06", 3.75),
    ("2003-07-10", 3.50), ("2003-11-06", 3.75), ("2004-02-05", 4.00), ("2004-05-06", 4.25),
    ("2004-06-10", 4.50), ("2004-08-05", 4.75), ("2005-08-04", 4.50), ("2006-08-03", 4.75),
    ("2006-11-09", 5.00), ("2007-01-11", 5.25), ("2007-05-10", 5.50), ("2007-07-05", 5.75),
    ("2007-12-06", 5.50), ("2008-02-07", 5.25), ("2008-04-10", 5.00), ("2008-10-08", 4.50),
    ("2008-11-06", 3.00), ("2008-12-04", 2.00), ("2009-01-08", 1.50), ("2009-02-05", 1.00),
    ("2009-03-05", 0.50), ("2016-08-04", 0.25), ("2017-11-02", 0.50), ("2018-08-02", 0.75),
    ("2020-03-11", 0.25), ("2020-03-19", 0.10), ("2021-12-16", 0.25), ("2022-02-03", 0.50),
    ("2022-03-17", 0.75), ("2022-05-05", 1.00), ("2022-06-16", 1.25), ("2022-08-04", 1.75),
    ("2022-09-22", 2.25), ("2022-11-03", 3.00), ("2022-12-15", 3.50), ("2023-02-02", 4.00),
    ("2023-03-23", 4.25), ("2023-05-11", 4.50), ("2023-06-22", 5.00), ("2023-08-03", 5.25),
    ("2024-08-01", 5.00), ("2024-11-07", 4.75),
]


def _bank_rate_at(dates):
    """Ставка, действовавшая на конкретную дату (последнее изменение не позже этой даты)."""
    ch = pd.DataFrame(BANK_RATE_CHANGES, columns=["date", "rate"])
    ch["date"] = pd.to_datetime(ch["date"])
    q = pd.DataFrame({"date": pd.to_datetime(dates)}).reset_index().sort_values("date")
    m = pd.merge_asof(q, ch, on="date", direction="backward").set_index("index").sort_index()
    return m["rate"].to_numpy()


def add_bank_rate(df):
    """bank_rate_pct - ставка на 1-е число месяца продажи (известна заранее, утечки нет);
    bank_rate_chg_12m - изменение за предыдущие 12 месяцев (пункты)."""
    out = df.copy()
    d = pd.to_datetime(dict(year=out["sale_year"], month=out["sale_month"], day=1))
    out["bank_rate_pct"] = _bank_rate_at(d)
    out["bank_rate_chg_12m"] = out["bank_rate_pct"] - _bank_rate_at(d - pd.DateOffset(years=1))
    return out


# ---------------------------------------------------------------------------
# Группа 3. Уровень цен в районе за предыдущий календарный год
# Источник: только train.csv соревнования Kaggle "London House Price Prediction: Advanced Techniques"
# https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques
# (внешних данных нет: медианы считаются по ценам продаж из этой же таблицы).
# ---------------------------------------------------------------------------
def _clean_hist(hist):
    h = hist[["outcode", "postcode", "sale_year", "price", "floorAreaSqM"]].copy()
    h["sector"] = h["postcode"].str[:-2].str.strip()  # 'E1 3AD' -> 'E1 3' (почтовый сектор)
    h["lp"] = np.log(h["price"])
    ok = h["floorAreaSqM"].between(15, 400)
    h["lpp"] = np.where(ok, np.log(h["price"] / h["floorAreaSqM"].where(ok)), np.nan)
    return h


def add_neighbourhood_history(df, hist, min_sector_n=5):
    """
    Для строки года Y берутся продажи ТОЛЬКО за год Y-1 в том же районе:
      oc_prev_med_logprice / oc_prev_med_logppsm - медианы log(цена) и log(цена за м²) по outcode,
      oc_prev_n                                  - число продаж в outcode за тот год,
      oc_price_yoy                               - изменение медианы log(цена) Y-1 к Y-2 (импульс рынка),
      sec_prev_med_logppsm / sec_prev_n          - то же по почтовому сектору (если продаж >= min_sector_n).
    Один и тот же лаг для train и test (для test год Y-1 = 2023) защищает от утечки и от сдвига
    между обучающей и тестовой выборкой. Год продажи выбран, а не скользящее окно, именно ради этого.
    """
    h = _clean_hist(hist)
    oc = h.groupby(["outcode", "sale_year"]).agg(med_lp=("lp", "median"), med_lpp=("lpp", "median"),
                                                  n=("lp", "size")).reset_index()
    sec = h.groupby(["sector", "sale_year"]).agg(med_lpp=("lpp", "median"), n=("lp", "size")).reset_index()
    sec.loc[sec["n"] < min_sector_n, "med_lpp"] = np.nan

    out = df.copy()
    key = out[["outcode"]].assign(sale_year=out["sale_year"] - 1)
    m1 = key.merge(oc, on=["outcode", "sale_year"], how="left")
    key2 = out[["outcode"]].assign(sale_year=out["sale_year"] - 2)
    m2 = key2.merge(oc, on=["outcode", "sale_year"], how="left")
    out["oc_prev_med_logprice"] = m1["med_lp"].to_numpy()
    out["oc_prev_med_logppsm"] = m1["med_lpp"].to_numpy()
    out["oc_prev_n"] = m1["n"].fillna(0).astype(int).to_numpy()
    out["oc_price_yoy"] = (m1["med_lp"] - m2["med_lp"]).to_numpy()
    ks = pd.DataFrame({"sector": out["postcode"].str[:-2].str.strip(), "sale_year": out["sale_year"] - 1})
    ms = ks.merge(sec, on=["sector", "sale_year"], how="left")
    out["sec_prev_med_logppsm"] = ms["med_lpp"].to_numpy()
    out["sec_prev_n"] = ms["n"].fillna(0).astype(int).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Группа 4. Предыдущая продажа того же адреса
# Источник: только train.csv соревнования Kaggle (ссылка выше, группа 3): одинаковый fullAddress
# в разные месяцы = повторная продажа. Внешних данных нет.
# ---------------------------------------------------------------------------
def add_repeat_sale(df, hist):
    """
    prev_sale_price       - цена предыдущей продажи по тому же fullAddress,
    prev_sale_gap_months  - сколько месяцев прошло с неё,
    n_prev_sales          - сколько раз адрес продавался раньше,
    prev_price_adj        - предыдущая цена, приведённая к уровню цен района (медиана outcode за год Y-1
                            делится на медиану за год предыдущей продажи).
    Для каждой строки df берётся последняя продажа того же адреса из hist СТРОГО раньше месяца этой строки.
    Одно и то же правило для train (df и hist - одна таблица) и для test (hist - весь train).
    Продажи того же месяца не считаются прошлыми (защита от утечки).
    """
    out = df.copy()
    t_out = out["sale_year"] * 12 + out["sale_month"]
    h = hist[["fullAddress", "sale_year", "sale_month", "price"]].copy()
    h["prev_t"] = h["sale_year"] * 12 + h["sale_month"]
    h = h.sort_values(["fullAddress", "prev_t"], kind="stable")
    h["n_prev"] = h.groupby("fullAddress").cumcount() + 1  # сколько продаж адреса было до этой включительно
    h = h.sort_values("prev_t", kind="stable")
    q = pd.DataFrame({"fullAddress": out["fullAddress"].to_numpy(), "t": t_out.to_numpy(),
                      "_row": np.arange(len(out))}).sort_values("t", kind="stable")
    m = pd.merge_asof(q, h[["fullAddress", "prev_t", "price", "n_prev"]], left_on="t", right_on="prev_t",
                      by="fullAddress", direction="backward", allow_exact_matches=False)
    m = m.sort_values("_row")
    out["prev_sale_price"] = m["price"].to_numpy()
    out["prev_sale_gap_months"] = (m["t"] - m["prev_t"]).to_numpy()
    out["n_prev_sales"] = m["n_prev"].fillna(0).astype(int).to_numpy()
    # приведение к уровню цен района
    hh = _clean_hist(hist)
    med = hh.groupby(["outcode", "sale_year"])["lp"].median().rename("med").reset_index()
    prev_year = ((t_out - 1 - out["prev_sale_gap_months"]) // 12)  # год предыдущей продажи
    now = out[["outcode"]].assign(sale_year=out["sale_year"] - 1).merge(med, on=["outcode", "sale_year"], how="left")["med"]
    then = out[["outcode"]].assign(sale_year=prev_year.to_numpy()).merge(med, on=["outcode", "sale_year"], how="left")["med"]
    out["prev_price_adj"] = out["prev_sale_price"] * np.exp(now.to_numpy() - then.to_numpy())
    return out


# ---------------------------------------------------------------------------
# Группа 5. Транспорт (станции)
# Источник: OpenStreetMap contributors (лицензия ODbL, https://www.openstreetmap.org/copyright),
# выгрузка станций через Overpass API (https://overpass-api.de) скриптом src/fetch_external.py (подкоманда stations).
# Теги: https://wiki.openstreetmap.org/wiki/Key:railway
# Сверка/дополнение: открытые данные TfL https://tfl.gov.uk/info-for/open-data-users/
# Файл data/external/stations_london.csv создаётся этим скриптом локально, в git не хранится.
# ---------------------------------------------------------------------------
# Ручная правка выгрузки OSM (проверено по списку станций 07.11.2026).
# 1) Объекты с тегом railway=station, которые не являются станциями городского транспорта:
#    выходы к самолётам и внутренний поезд аэропорта Хитроу, детские и парковые узкоколейки.
NOT_STATIONS = {
    "Gates A1-A23", "Gates B32-B48", "Gates C52-C66", "Station A", "Station B",
    "Swanley New Barn Miniature Railway", "Eddie's Rest", "Powdermill Cut", "Long Walk", "Barking Park",
    "Willow Lawn", "Woody Bay", "Haste Hill", "Cassiobury Park Station", "Depot Approach", "Ealing End",
    "Hanworth Halt",
}
# 2) Станции, открытые после 1995 года (начало данных), у которых в OSM нет start_date.
#    Без года такая станция считается открытой всегда, и продажа 1996 года "видит" станцию,
#    которой ещё нет. Годы открытия: статьи о станциях и линиях (Jubilee line extension 1999,
#    DLR, London Overground, Elizabeth line 2022, Northern line extension 2021).
#    Ключ: (название, kind), потому что одно название бывает у станций разных видов.
OPEN_YEAR_FIX = {
    # продление Jubilee line, 1999
    ("Southwark", "tube"): 1999, ("Bermondsey", "tube"): 1999, ("Canada Water", "tube"): 1999,
    ("North Greenwich", "tube"): 1999, ("Canary Wharf", "tube"): 1999,
    # DLR: Pudding Mill Lane 1996 (в OSM указан 2014, год переноса станции), Lewisham 1999,
    # London City Airport 2005, Langdon Park 2007, Stratford International 2011
    ("Pudding Mill Lane", "other"): 1996,
    ("Cutty Sark for Maritime Greenwich", "other"): 1999, ("Deptford Bridge", "other"): 1999,
    ("Elverson Road", "other"): 1999, ("Lewisham DLR", "other"): 1999,
    ("West Silvertown", "other"): 2005, ("Pontoon Dock", "other"): 2005,
    ("London City Airport", "other"): 2005, ("King George V", "other"): 2005,
    ("Langdon Park", "other"): 2007,
    ("Stratford High Street", "other"): 2011, ("Abbey Road", "other"): 2011, ("Star Lane", "other"): 2011,
    ("Stratford International DLR", "other"): 2011,
    # London Overground и другие
    ("Wood Lane", "tube"): 2008, ("Shepherd's Bush", "rail"): 2008, ("Imperial Wharf", "rail"): 2009,
    ("Stratford International", "rail"): 2009,
    ("Shoreditch High Street", "rail"): 2010, ("Hoxton", "rail"): 2010, ("Haggerston", "rail"): 2010,
    ("Dalston Junction", "rail"): 2010,
    # продление Northern line, 2021
    ("Nine Elms", "tube"): 2021, ("Battersea Power Station", "tube"): 2021,
    # Elizabeth line и Barking Riverside, 2022
    ("Woolwich", "rail"): 2022, ("Custom House", "rail"): 2022, ("Canary Wharf", "rail"): 2022,
    ("Bond Street", "rail"): 2022, ("Tottenham Court Road", "rail"): 2022, ("Barking Riverside", "rail"): 2022,
}


def load_stations(path):
    """CSV: name, lat, lon, kind ('tube' | 'rail' | 'other'), open_year (может быть пустым).
    Применяет ручную правку NOT_STATIONS и OPEN_YEAR_FIX и печатает, что изменилось."""
    st = pd.read_csv(path)
    need = {"name", "lat", "lon", "kind"}
    if not need.issubset(st.columns):
        sys.exit(f"В {path} нет столбцов {need - set(st.columns)}")
    if "open_year" not in st.columns:
        st["open_year"] = np.nan
    st = st.dropna(subset=["lat", "lon"])
    drop = st["name"].isin(NOT_STATIONS)
    st = st[~drop].reset_index(drop=True)
    fixed = 0
    for (name, kind), year in OPEN_YEAR_FIX.items():
        m = (st["name"] == name) & (st["kind"] == kind)
        if not m.any():
            print(f"Станции: не найдена {name} ({kind}) из OPEN_YEAR_FIX")
        fixed += int(m.sum())
        st.loc[m, "open_year"] = year
    print(f"Станции: {len(st)} после правки (убрано {int(drop.sum())} не-станций, год открытия задан у {fixed})")
    return st


def _query(stations, lat, lon):
    from sklearn.neighbors import BallTree
    n = len(lat)
    if len(stations) == 0:
        return np.full(n, np.nan), np.zeros(n, dtype=int)
    tree = BallTree(np.radians(stations[["lat", "lon"]].to_numpy()), metric="haversine")
    pts = np.radians(np.c_[lat, lon])
    dist, _ = tree.query(pts, k=1)
    cnt = tree.query_radius(pts, r=1.0 / EARTH_R_KM, count_only=True)
    return dist[:, 0] * EARTH_R_KM, cnt


def add_transport(df, stations):
    """dist_station_km, dist_tube_km, n_stations_1km; учитываются станции с open_year <= sale_year
    (если год неизвестен, станция считается открытой всегда)."""
    out = df.copy()
    for c in ("dist_station_km", "dist_tube_km", "n_stations_1km"):
        out[c] = np.nan
    year_open = stations["open_year"].fillna(0)
    for year, idx in out.groupby("sale_year").groups.items():
        active = stations[year_open <= year]
        lat = out.loc[idx, "latitude"].to_numpy()
        lon = out.loc[idx, "longitude"].to_numpy()
        d_all, n_all = _query(active, lat, lon)
        d_tube, _ = _query(active[active["kind"] == "tube"], lat, lon)
        out.loc[idx, "dist_station_km"] = d_all
        out.loc[idx, "dist_tube_km"] = d_tube
        out.loc[idx, "n_stations_1km"] = n_all
    out["n_stations_1km"] = out["n_stations_1km"].astype(int)
    return out


# ---------------------------------------------------------------------------
# Группа 6. Локальный контекст: плотность застройки и цены ближайших продаж
# Источник: только train.csv и test.csv соревнования Kaggle (ссылка в начале файла).
# ---------------------------------------------------------------------------
def add_local_context(df, hist, addresses, radius_km=0.5, k=20):
    """
    n_addr_500m              - число других адресов из train+test в радиусе 500 м. Это ПРИБЛИЗИТЕЛЬНАЯ мера
                               плотности застройки: данные - выборка продаж, а не полный реестр домов.
                               Цены не используются, поэтому утечки нет.
    knn20_prev_med_logppsm   - медиана log(цена за м²) 20 ближайших продаж ЗА ПРЕДЫДУЩИЙ год,
    knn20_prev_dist_km       - расстояние до самой дальней из этих 20 продаж (чем больше, тем реже рынок).
    Тот же лаг в 1 календарный год, что в группе 3, поэтому утечки и сдвига между train и test нет.
    """
    from sklearn.neighbors import BallTree
    out = df.copy()
    # плотность
    a = addresses.drop_duplicates("fullAddress")
    tree = BallTree(np.radians(a[["latitude", "longitude"]].to_numpy()), metric="haversine")
    pts = np.radians(out[["latitude", "longitude"]].to_numpy())
    cnt = tree.query_radius(pts, r=radius_km / EARTH_R_KM, count_only=True)
    out["n_addr_500m"] = np.maximum(cnt - 1, 0)  # минус сам объект
    # цены ближайших продаж прошлого года
    h = _clean_hist(hist.assign(latitude=hist["latitude"], longitude=hist["longitude"]))
    h["latitude"], h["longitude"] = hist["latitude"].to_numpy(), hist["longitude"].to_numpy()
    out["knn20_prev_med_logppsm"] = np.nan
    out["knn20_prev_dist_km"] = np.nan
    for year, idx in out.groupby("sale_year").groups.items():
        src = h[(h["sale_year"] == year - 1) & h["lpp"].notna()]
        if len(src) < k:
            continue
        t = BallTree(np.radians(src[["latitude", "longitude"]].to_numpy()), metric="haversine")
        dist, ind = t.query(np.radians(out.loc[idx, ["latitude", "longitude"]].to_numpy()), k=k)
        out.loc[idx, "knn20_prev_med_logppsm"] = np.median(src["lpp"].to_numpy()[ind], axis=1)
        out.loc[idx, "knn20_prev_dist_km"] = dist[:, -1] * EARTH_R_KM
    return out


# ---------------------------------------------------------------------------
# Группа 7. Район (borough) и LSOA по почтовому индексу
# Источник: postcodes.io (https://postcodes.io, документация https://api.postcodes.io/docs),
# данные ONS Postcode Directory. Таблица data/external/postcode_lookup.csv создаётся локально
# скриптом src/fetch_external.py, подкоманда postcodes (нужен интернет), в git не хранится.
# ---------------------------------------------------------------------------
def load_postcode_lookup(path):
    lk = pd.read_csv(path)
    need = {"postcode", "borough", "lsoa_code"}
    if not need.issubset(lk.columns):
        sys.exit(f"В {path} нет столбцов {need - set(lk.columns)}")
    return lk.drop_duplicates("postcode")


def add_postcode_area(df, lookup):
    """borough - лондонский район (33 значения, категориальный признак); lsoa_code - код small area ONS
    (нужен для группы 8, как признак модели не используется)."""
    out = df.copy()
    m = out[["postcode"]].merge(lookup[["postcode", "borough", "lsoa_code"]], on="postcode", how="left")
    out["borough"] = m["borough"].to_numpy()
    out["lsoa_code"] = m["lsoa_code"].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Группа 8. Индекс депривации IMD 2019 по LSOA
# Источник: English Indices of Deprivation 2019 (MHCLG), лицензия Open Government Licence v3.
# Файл "ID 2019 for London.xlsx" на London Datastore:
# https://data.london.gov.uk/dataset/indices-of-deprivation
# (скачать вручную, положить в data/external/). IMD 2019 снимок на один год, а цены идут с 1995:
# это признак относительной "престижности" района, а не динамики.
# ---------------------------------------------------------------------------
IMD_PATTERNS = {  # столбец результата: (обязательные слова в названии, запрещённые слова)
    "imd_score": (["multiple deprivation", "score"], []),
    "imd_decile": (["multiple deprivation", "decile"], []),
    "crime_decile": (["crime", "decile"], []),
    "living_env_decile": (["living environment", "decile"], []),
    "income_decile": (["income", "decile"], ["children", "older", "deprivation affecting"]),
}


def load_imd(path):
    """Читает файл IMD 2019 (xlsx или csv) и возвращает таблицу lsoa_code + показатели.
    Названия столбцов в разных версиях файла отличаются, поэтому ищем их по ключевым словам
    и печатаем, что нашли. Если чего-то не нашлось, смотрите печать и поправьте IMD_PATTERNS."""
    sheets = pd.read_excel(path, sheet_name=None) if str(path).lower().endswith(("xlsx", "xls")) \
        else {"csv": pd.read_csv(path)}
    found = {}
    for sname, sh in sheets.items():
        cols = {str(c): str(c).lower() for c in sh.columns}
        code = next((c for c, l in cols.items() if "lsoa" in l and "code" in l), None)
        if code is None:
            continue
        for out_col, (must, bad) in IMD_PATTERNS.items():
            if out_col in found:
                continue
            c = next((c for c, l in cols.items() if all(w in l for w in must) and not any(w in l for w in bad)), None)
            if c is not None:
                found[out_col] = sh[[code, c]].rename(columns={code: "lsoa_code", c: out_col}).set_index("lsoa_code")
                print(f"IMD: {out_col} <- лист '{sname}', столбец '{c}'")
    missing = [k for k in IMD_PATTERNS if k not in found]
    if missing:
        print("IMD: НЕ найдено:", missing)
    if not found:
        sys.exit("В файле IMD не найден ни один нужный столбец")
    return pd.concat(found.values(), axis=1).reset_index()


def add_imd(df, imd):
    """Присоединяет показатели IMD по lsoa_code (нужна группа 7). Печатает долю совпавших кодов:
    она должна быть близка к 100%; если мала, версии LSOA в postcodes.io и в IMD (2011 и 2021) не совпали."""
    out = df.copy()
    m = out[["lsoa_code"]].merge(imd.drop_duplicates("lsoa_code"), on="lsoa_code", how="left")
    cols = [c for c in imd.columns if c != "lsoa_code"]
    for c in cols:
        out[c] = m[c].to_numpy()
    print(f"IMD: совпало {out[cols[0]].notna().mean():.1%} строк")
    return out


# ---------------------------------------------------------------------------
def build(train, test, stations=None, postcodes=None, imd=None, hist=None):
    """Возвращает (train_enriched, test_enriched).
    hist - продажи, по ценам которых считаются группы 3, 4, 6 (по умолчанию весь train).
    В build_dataset.py сюда передаётся train без дубликатов и подозрительных цен (см. clean.py)."""
    hist = train if hist is None else hist
    tr = add_centre_distance(train)
    te = add_centre_distance(test)
    tr, te = add_bank_rate(tr), add_bank_rate(te)
    tr = add_neighbourhood_history(tr, hist)
    te = add_neighbourhood_history(te, hist)
    tr = add_repeat_sale(tr, hist)
    te = add_repeat_sale(te, hist)
    addr = pd.concat([train, test])[["fullAddress", "latitude", "longitude"]]
    tr = add_local_context(tr, hist, addr)
    te = add_local_context(te, hist, addr)
    if stations is not None:
        tr, te = add_transport(tr, stations), add_transport(te, stations)
    if postcodes is not None:
        tr, te = add_postcode_area(tr, postcodes), add_postcode_area(te, postcodes)
        if imd is not None:
            tr, te = add_imd(tr, imd), add_imd(te, imd)
    return tr, te

