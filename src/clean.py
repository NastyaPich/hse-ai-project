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


# ---------------------------------------------------------------------------------------------------
# Чекпойнт 3: правила для моделей. Их вызывает ТОЛЬКО код моделей (src/ml_utils.py); основная база
# (build_dataset.py) их не использует и не меняется. Старые флаги выше по смыслу не меняются
# (на них опираются числа в EDA).
#
#   price_suspect_noarea   у объекта нет площади, поэтому правило price_suspect (цена за м²) к нему не применимо.
#                          Сравниваем цену с медианой цены похожих продаж (тот же outcode и год, тот же тип жилья
#                          и число спален; если таких меньше 5 - тот же outcode-год-тип, затем outcode-год,
#                          затем outcode за все годы). Флаг = цена отличается от медианы больше чем в NOAREA_RATIO раз.
#                          Порог 10: у обычных продаж с известной площадью (price_suspect = 0) так далеко от медианы
#                          лежит только 0,14% строк, а у подозрительных (price_suspect = 1) - 37%.
#   addr_multi_unit_past   как addr_multi_unit, но без взгляда в будущее: у адреса уже БЫЛИ две продажи с интервалом
#                          <= 3 месяцев, и обе раньше месяца текущей строки (только по истории hist).
#   history_mask_strict    как history_mask, но ещё без price_suspect_noarea.
# ---------------------------------------------------------------------------------------------------
NOAREA_RATIO = 10
NOAREA_KEYS = [['outcode', 'sale_year', 'pt_group_filled', 'bedrooms_filled'],
               ['outcode', 'sale_year', 'pt_group_filled'], ['outcode', 'sale_year'], ['outcode']]


def add_noarea_flag(train_filled, test_filled, ratio=NOAREA_RATIO, min_n=5):
    """Дописывает столбец price_suspect_noarea (нужны флаги add_clean_flags и столбцы fill_missing).
    Медианы считаются по train без дубликатов и подозрительных цен. В test цены нет, флаг = 0."""
    tr, te = train_filled.copy(), test_filled.copy()
    ref = tr[history_mask(tr)].assign(_lp=lambda d: np.log(d['price']))
    med = pd.Series(np.nan, index=tr.index)
    for k in NOAREA_KEYS:
        g = ref.groupby(k)['_lp'].agg(['median', 'size'])
        g = g.loc[g['size'] >= min_n, 'median'].rename('_m')
        med = med.fillna(tr[k].join(g, on=k)['_m'])
    dev = (np.log(tr['price']) - med).abs()
    tr['price_suspect_noarea'] = (tr['floorAreaSqM'].isna() & (dev > np.log(ratio))).astype(int)
    te['price_suspect_noarea'] = 0
    return tr, te


def history_mask_strict(train_flagged):
    """Как history_mask, но дополнительно без price_suspect_noarea (если такой столбец есть)."""
    m = history_mask(train_flagged)
    if 'price_suspect_noarea' in train_flagged.columns:
        m = m & (train_flagged['price_suspect_noarea'] == 0)
    return m


def addr_multi_unit_past(df, hist, months=MULTI_UNIT_MONTHS):
    """0/1 для каждой строки df: среди продаж hist того же адреса СТРОГО раньше месяца строки уже есть пара
    с интервалом <= months месяцев. Используются только даты (не цены). hist - прошлые продажи
    (для фолда Y - train до Y-1, для test - весь train); полные дубликаты из hist убираются."""
    h = hist
    if 'is_duplicate' in h.columns:
        h = h[h['is_duplicate'] == 0]
    h = h[['fullAddress', 'sale_year', 'sale_month']].assign(t=h['sale_year'] * 12 + h['sale_month'])
    h = h.sort_values(['fullAddress', 't'])
    gap = h.groupby('fullAddress')['t'].diff()
    first = h.loc[gap <= months].groupby('fullAddress')['t'].min()  # месяц, когда пара впервые «сложилась»
    t = df['sale_year'] * 12 + df['sale_month']
    return (df['fullAddress'].map(first) < t).fillna(False).astype(int)


def summary(tr, te):
    """Таблица: сколько строк помечено каждым флагом."""
    rows = []
    for c in FLAGS[1:]:
        rows.append({'флаг': c, 'train, строк': int(tr[c].sum()), 'train, %': tr[c].mean() * 100,
                     'test, строк': int(te[c].sum()), 'test, %': te[c].mean() * 100})
    return pd.DataFrame(rows).set_index('флаг').round(2)
