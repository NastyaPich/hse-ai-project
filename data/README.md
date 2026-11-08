# Данные

Сами таблицы в git не хранятся (кроме архива соревнования, он нужен ноутбуку EDA). Ниже показано, что где должно лежать. Все команды запускаются из корня репозитория.

```
data/
├── raw/
│   ├── london-house-price-prediction-advanced-techniques.zip   # архив Kaggle (в git)
│   └── london/            # распакованные train.csv, test.csv (в git не храним)
├── external/              # внешние данные для признаков (в git не храним)
│   ├── stations_london.csv        # станции, src/fetch_stations_osm.py
│   ├── postcode_lookup.csv        # район и LSOA, src/fetch_postcode_lookup.py
│   └── ID 2019 for London.xlsx    # IMD 2019, скачать вручную
└── processed/             # результаты скриптов (в git не храним)
    ├── train_enriched.csv, test_enriched.csv   # src/build_dataset.py: все признаки и флаги
    └── train_final.csv, test_final.csv         # src/build_dataset.py: итоговый набор
```

## Откуда берутся данные

| Файл | Источник | Как получить |
|------|----------|--------------|
| `train.csv`, `test.csv` | [Kaggle: London House Price Prediction: Advanced Techniques](https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques) | распаковать архив: `unzip data/raw/london-house-price-prediction-advanced-techniques.zip -d data/raw/london` |
| `stations_london.csv` | OpenStreetMap (ODbL) через Overpass API | `python src/fetch_stations_osm.py` (нужен интернет) |
| `postcode_lookup.csv` | [postcodes.io](https://postcodes.io) (ONS Postcode Directory) | `python src/fetch_postcode_lookup.py` (нужен интернет) |
| `ID 2019 for London.xlsx` | English Indices of Deprivation 2019 (OGL v3), [London Datastore](https://data.london.gov.uk/dataset/indices-of-deprivation) | скачать вручную и положить в `data/external/` |

Подробно о признаках, источниках и защите от утечек: [`FEATURES.md`](../FEATURES.md).
