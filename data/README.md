# CarValue AI Used-Car Dataset

## 1. Overview

This dataset contains used-car listings collected from **Bonbanh.com**. Each row represents one listing and includes vehicle identity, technical specifications, asking price, and seller location.

The dataset supports two main tasks:

- Exploring the used-car market by brand, body type, production year, mileage, origin, and location.
- Building a model to estimate a vehicle's **listed asking price** from its observed attributes.

> `price_vnd` is the price shown in the online listing, not the final transaction price.

## 2. Project files

| File | Description |
|---|---|
| `bonbanh_usedcar_dataset (3)(1).csv` | Main dataset used for analysis, containing 23,187 rows and 14 columns |
| `bonbanh(1).ipynb` | Data collection and regional file standardization code |
| `CarValue_AI_Research_EDA_Master_executed (1).ipynb` | Data cleaning, exploratory analysis, and modeling preparation |

The CSV file uses UTF-8 encoding and is approximately 4.23 MB.

## 3. Data collection

The data was collected by region with `requests` and `BeautifulSoup`. The notebook uses the following process:

1. Select a regional used-car results page and define its `location` label and output filename.
2. Visit each results page and collect listing URLs. A Python `set` removes repeated URLs, and the link list is saved as a backup CSV.
3. Open each vehicle detail page and extract the title and specifications, including year, mileage, origin, body type, transmission, engine, color, seats, and drivetrain.
4. Extract the price, clean the vehicle name, derive `brand` and `model`, add the location, align the columns, and export the regional CSV.

The notebook uses a 20-second timeout for results pages, a 15-second timeout for detail pages, and a one-second delay between requests. If a request fails during URL collection, the script waits five seconds before retrying. Failed detail pages are retained with their URLs for later inspection.

## 4. Dataset size and scope

| Property | Value |
|---|---:|
| Rows | 23,187 |
| Columns | 14 |
| Brands | 80 |
| Distinct raw model strings | 5,117 |
| Production years | 1989–2026 |
| Locations with recorded values | 62 |
| Median available asking price | 545,000,000 VND |
| Observed price range | 16,000,000–66,000,000,000 VND |

This is a cross-sectional snapshot from one marketplace. The dataset has no reliable listing timestamp, so it should not be used to measure price changes over time.

## 5. Data dictionary

| Column | Current type | Description | Example |
|---|---|---|---|
| `brand` | text | Vehicle manufacturer | `Toyota`, `BMW` |
| `model` | text | Model and variant; some malformed values still contain the year and price | `A5 Sportback 2.0` |
| `year` | number | Production year | `2017` |
| `price_vnd` | number | Listed asking price in VND | `799000000` |
| `mileage_km` | text | Reported mileage, including commas and the `Km` suffix | `70,000 Km` |
| `condition` | text | Vehicle condition | `Xe đã dùng` |
| `origin` | text | Imported or domestically assembled | `Nhập khẩu` |
| `body_type` | text | Vehicle body type | `SUV`, `Sedan` |
| `transmission` | text | Transmission type | `Số tự động` |
| `engine` | text | Fuel type and engine displacement | `Xăng 2.0 L` |
| `exterior_color` | text | Exterior color | `Đen` |
| `seats` | text | Number of seats | `5 chỗ` |
| `drivetrain` | text | Drivetrain configuration | `FWD - Dẫn động cầu trước` |
| `location` | text | Province or city associated with the listing | `Hà Nội` |

Vietnamese category values are preserved because they come directly from the source listings.

## 6. Data quality issues

### Missing prices

`price_vnd` is missing in 3,911 rows, or 16.87% of the dataset. In many of these rows, the price is still embedded in `model`, for example:

```text
DB11 4.0 V8 2021 - 9 Tỷ 999 Triệu
```

The value can be recovered with regular expressions covering `tỷ`, `tỉ`, `triệu`, and `tr`. A `price_recovered_flag` should be retained to distinguish recovered prices from original values.

### Missing locations

`location` is missing in 9,444 rows, or 40.73%. Missingness may be associated with collection batches rather than the underlying market. Location should be used cautiously for exploratory analysis and excluded from the baseline model until its coverage is verified.

### Mileage stored as text

`mileage_km` is stored as text. There are 3,249 rows recorded as `0 Km`, and the data contains extreme values such as 3,380,000,000 km. A zero value should not automatically be treated as the true mileage of a used vehicle. A practical approach is to convert it to missing, create `mileage_zero_flag`, and inspect outliers after numeric conversion.

### Constant condition field

All 23,187 rows have `condition = Xe đã dùng`. This field does not distinguish listings and should be removed from the model inputs.

### Fully identical rows

The file contains 1,551 rows that are identical across all 14 columns. However, the final dataset does not retain a listing ID or URL. Identical rows may represent different vehicles, reposted listings, or true duplicates. They should not be removed automatically without a reliable listing identifier.

## 7. Recommended preprocessing

1. Recover missing `price_vnd` values from price expressions embedded in `model`, and retain a recovery flag.
2. Remove embedded prices and year suffixes from `model`; standardize known brand and model spelling errors.
3. Convert `year`, `price_vnd`, `mileage_km`, `seats`, and engine displacement to numeric values.
4. Split `engine` into fuel type and engine displacement.
5. Replace placeholder values such as `-` with missing values.
6. Inspect implausible years, extreme mileage values, and price outliers.
7. Create `car_age = reference_year - year` after defining a clear reference year.
8. Use group-aware train/test splitting for records with the same vehicle signature to reduce data leakage.

## 8. Loading the data with Python

```python
import pandas as pd

df = pd.read_csv(
    "bonbanh_usedcar_dataset (3)(1).csv",
    encoding="utf-8"
)

print(df.shape)
print(df.head())
print(df.isna().sum())
```

Example mileage conversion:

```python
df["mileage_km_num"] = pd.to_numeric(
    df["mileage_km"]
      .str.replace(",", "", regex=False)
      .str.extract(r"(\d+(?:\.\d+)?)")[0],
    errors="coerce"
)
```

## 9. Interpretation limits

- The dataset represents listings on Bonbanh.com and may not represent the entire Vietnamese used-car market.
- Prices are seller asking prices, not completed transaction prices.
- The absence of reliable listing dates prevents time-trend, inflation, and policy-impact analysis.
- Comparisons by brand, origin, location, or body type describe associations within this sample and should not be interpreted as causal effects.
- The absence of listing IDs makes duplicate identification uncertain.

## 10. Source

- Website: [Bonbanh.com](https://bonbanh.com/)
- Collection notebook: `bonbanh(1).ipynb`
- Dataset checked: `bonbanh_usedcar_dataset (3)(1).csv`

