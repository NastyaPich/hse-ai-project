"""
Очистка данных по правилам из EDA (notebooks/EDA.ipynb, разделы 3-5).

Принцип: строки НЕ удаляем, а помечаем флагами (0/1). Что выкидывать при обучении,
решаем на этапе моделей, и это решение остаётся видимым и обратимым.
Исключение: признаки, которые считаются по ценам (цены района, прошлая продажа, цены соседей),
считаются только по "чистой" истории (history_mask), иначе одна ошибочная цена испортит
признаки у всех соседей.

Флаги:
  pt_group          тип жилья, 19 названий сведены к 6 группам (EDA, раздел 4.1)
  is_duplicate      полный дубликат строки (все столбцы, кроме ID, совпадают); первая копия не помечается
  dup_addr_month    у адреса несколько продаж в одном месяце с разной ценой (опечатка или разные квартиры)
  price_suspect     цена за м² меньше 300 £ или больше 40 000 £ (EDA, раздел 5); только train
  area_suspect      площадь ровно 10 м² или 499 м² и больше (похоже на обрезку значений)
  addr_multi_unit   под одним адресом, видимо, несколько квартир: у адреса есть две продажи
                    с интервалом не больше 3 месяцев (настоящая перепродажа так быстро почти не бывает)

Заполнение пропусков (fill_missing, правила из EDA, раздел 3): исходные столбцы не меняются,
заполненные значения пишутся в новые столбцы с суффиксом _filled, рядом флаги <столбец>_missing.
  floorAreaSqM      медиана по типу жилья (pt_group) и числу спален; если спален нет - по типу жилья
  bedrooms, bathrooms, livingRooms   медиана по типу жилья
  tenure, currentEnergyRating, pt_group   категория 'Unknown'
Медианы считаются только по train (без подозрительных площадей), test их не видит.
"""
import numpy as np
import pandas as pd

PT_GROUP = {
    'Purpose Built Flat': 'Flat', 'Flat/Maisonette': 'Flat', 'Converted Flat': 'Flat',
    'Mid Terrace House': 'Terraced', 'Terrace Property': 'Terraced', 'Terraced': 'Terraced',
    'Mid Terrace Property': 'Terraced',
    'End Terrace House': 'End terrace', 'End Terrace Property': 'End terrace',
    'Semi-Detached House': 'Semi-detached', 'Semi-Detached Property': 'Semi-detached',
    'Detached House': 'Detached', 'Detached Property': 'Detached',
    'Bungalow Property': 'Bungalow', 'Semi-Detached Bungalow': 'Bungalow', 'Detached Bungalow': 'Bungalow',
    'Mid Terrace Bungalow': 'Bungalow', 'Terraced Bungalow': 'Bungalow', 'End Terrace Bungalow': 'Bungalow',
}
PPSM_LOW, PPSM_HIGH = 300, 40_000      # границы подозрительной цены за м², £
AREA_LOW, AREA_HIGH = 10, 499          # подозрительная площадь: == 10 или >= 499 м²
MULTI_UNIT_MONTHS = 3                  # две продажи адреса ближе этого срока -> адрес на несколько квартир

FLAGS = ['pt_group', 'is_duplicate', 'dup_addr_month', 'price_suspect', 'area_suspect', 'addr_multi_unit']
NUM_FILL = ['floorAreaSqM', 'bedrooms', 'bathrooms', 'livingRooms']
CAT_FILL = ['tenure', 'currentEnergyRating', 'pt_group']
FILLED = ([f'{c}_filled' for c in NUM_FILL] + [f'{c}_missing' for c in NUM_FILL]
          + [f'{c}_filled' for c in CAT_FILL])


def _multi_unit_addresses(frames):
    """Адреса, у которых есть две продажи с интервалом <= MULTI_UNIT_MONTHS месяцев.
    Используются только даты продаж (не цены), поэтому train и test можно смотреть вместе."""
    # полные дубликаты (одна и та же запись дважды) убираем, иначе они дают ложный интервал 0;
    # несколько разных продаж в одном месяце (интервал 0) оставляем: это как раз признак многоквартирного адреса
    d = pd.concat([f.drop_duplicates(f.columns.drop('ID'))[['fullAddress', 'sale_year', 'sale_month']]
                   for f in frames])
    t = d['sale_year'] * 12 + d['sale_month']
    d = d.assign(t=t).sort_values(['fullAddress', 't'])
    gap = d.groupby('fullAddress')['t'].diff()
    return set(d.loc[gap <= MULTI_UNIT_MONTHS, 'fullAddress'])


def add_clean_flags(train, test):
    """Возвращает копии train и test с дописанными справа столбцами FLAGS."""
    tr, te = train.copy(), test.copy()
    for d in (tr, te):
        d['pt_group'] = d['propertyType'].map(PT_GROUP)
        d['is_duplicate'] = d.drop(columns='ID').duplicated().astype(int)
        key = ['fullAddress', 'sale_year', 'sale_month']
        d['dup_addr_month'] = 0
        if 'price' in d.columns:
            n_prices = d.groupby(key)['price'].transform('nunique')
            d['dup_addr_month'] = (n_prices > 1).astype(int)
            ppsm = d['price'] / d['floorAreaSqM']
            d['price_suspect'] = ((ppsm < PPSM_LOW) | (ppsm > PPSM_HIGH)).astype(int)
        else:
            d['price_suspect'] = 0  # в test цены нет
        a = d['floorAreaSqM']
        d['area_suspect'] = ((a == AREA_LOW) | (a >= AREA_HIGH)).astype(int)
    multi = _multi_unit_addresses([tr, te])
    for d in (tr, te):
        d['addr_multi_unit'] = d['fullAddress'].isin(multi).astype(int)
    return tr, te


def fill_missing(train_flagged, test_flagged):
    """Возвращает копии train и test с дописанными столбцами FILLED (нужен pt_group из add_clean_flags)."""
    tr, te = train_flagged.copy(), test_flagged.copy()
    ref = tr[tr['area_suspect'] == 0].assign(_pt=tr['pt_group'].fillna('Unknown'))
    glob = ref[NUM_FILL].median()
    by_type = ref.groupby('_pt')[NUM_FILL].median()
    area_type_bed = ref.groupby(['_pt', 'bedrooms'])['floorAreaSqM'].median().rename('_a_tb')
    for d in (tr, te):
        pt = d['pt_group'].fillna('Unknown')
        for c in NUM_FILL:
            d[f'{c}_missing'] = d[c].isna().astype(int)
        # площадь: тип + спальни -> тип -> общая медиана
        a_tb = pd.DataFrame({'_pt': pt, 'bedrooms': d['bedrooms']}).join(area_type_bed, on=['_pt', 'bedrooms'])['_a_tb']
        d['floorAreaSqM_filled'] = (d['floorAreaSqM'].fillna(a_tb).fillna(pt.map(by_type['floorAreaSqM']))
                                    .fillna(glob['floorAreaSqM']))
        # комнаты: тип -> общая медиана
        for c in ['bedrooms', 'bathrooms', 'livingRooms']:
            d[f'{c}_filled'] = d[c].fillna(pt.map(by_type[c])).fillna(glob[c])
        for c in CAT_FILL:
            d[f'{c}_filled'] = d[c].fillna('Unknown')
    return tr, te


def history_mask(train_flagged):
    """Строки train, по которым можно считать признаки из цен: без дубликатов и подозрительных цен."""
    return (train_flagged['is_duplicate'] == 0) & (train_flagged['price_suspect'] == 0)


def summary(tr, te):
    """Таблица: сколько строк помечено каждым флагом."""
    rows = []
    for c in FLAGS[1:]:
        rows.append({'флаг': c, 'train, строк': int(tr[c].sum()), 'train, %': tr[c].mean() * 100,
                     'test, строк': int(te[c].sum()), 'test, %': te[c].mean() * 100})
    return pd.DataFrame(rows).set_index('флаг').round(2)
