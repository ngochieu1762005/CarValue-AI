import csv
import re
import time
import random
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

CATEGORY_PAGES = [
    "https://bonbanh.com/da-nang/oto",
    "https://bonbanh.com/da-nang/oto/toyota",
    "https://bonbanh.com/da-nang/oto/hyundai",
    "https://bonbanh.com/da-nang/oto/kia",
    "https://bonbanh.com/da-nang/oto/mazda",
    "https://bonbanh.com/da-nang/oto/ford",
    "https://bonbanh.com/da-nang/oto/honda",
]

PAGES_PER_CATEGORY = 6
TARGET_ROWS = 110
REQUEST_DELAY = (0.6, 1.2)
LOCATION = "Đà Nẵng"

SPEC_LABELS = {
    "Năm sản xuất": "year",
    "Tình trạng": "condition",
    "Xuất xứ": "origin",
    "Kiểu dáng": "body_type",
    "Hộp số": "transmission",
    "Động cơ": "engine",
    "Màu ngoại thất": "exterior_color",
    "Số chỗ ngồi": "seats",
    "Dẫn động": "drivetrain",
    "Số km đã đi": "mileage_km",
    "Số km": "mileage_km",
}

FIELDNAMES = [
    "brand",
    "model",
    "year",
    "price_vnd",
    "mileage_km",
    "condition",
    "origin",
    "body_type",
    "transmission",
    "engine",
    "exterior_color",
    "seats",
    "drivetrain",
    "location",
]


def extract_detail_links(html):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.select("a[itemprop='url']"):
        href = a.get("href", "")
        if re.search(r"-\d{5,}$", href):
            links.add(urljoin("https://bonbanh.com/", href))
    return links


def collect_detail_urls():
    urls = []
    for cat in CATEGORY_PAGES:
        for page_num in range(1, PAGES_PER_CATEGORY + 1):
            page_url = cat if page_num == 1 else f"{cat}/{page_num}"
            try:
                resp = requests.get(page_url, headers=HEADERS, timeout=15)
                resp.raise_for_status()
                resp.encoding = "utf-8"
                html = resp.text
            except requests.RequestException as e:
                print(f"  skip {page_url}: {e}")
                continue

            new_links = extract_detail_links(html)
            for link in new_links:
                if link not in urls:
                    urls.append(link)
            print(f"  {page_url}: +{len(new_links)} found (total unique {len(urls)})")
            time.sleep(random.uniform(*REQUEST_DELAY))

            if len(urls) >= TARGET_ROWS:
                return list(dict.fromkeys(urls))
    return list(dict.fromkeys(urls))


def extract_brand_model(soup):
    brand, model = "", ""
    for item in soup.select("span[itemprop='itemListElement']"):
        pos_tag = item.select_one("meta[itemprop='position']")
        name_tag = item.select_one("span[itemprop='name']")
        if not pos_tag or not name_tag:
            continue
        pos = pos_tag.get("content")
        text = name_tag.get_text(strip=True)
        if pos == "3":
            brand = text
        elif pos == "4":
            model = text
    return brand, model


def extract_price_vnd(soup, html):
    price_tag = soup.select_one("[itemprop='price']")
    if price_tag and price_tag.get("content", "").isdigit():
        return int(price_tag["content"])
    m_ty = re.search(r"(\d+)\s*Tỷ", html)
    m_trieu = re.search(r"(\d+)\s*Triệu", html)
    total = 0
    if m_ty:
        total += int(m_ty.group(1)) * 1_000_000_000
    if m_trieu:
        total += int(m_trieu.group(1)) * 1_000_000
    return total or None


def normalize_seats(text):
    m = re.search(r"\d+", text or "")
    return m.group(0) if m else ""


def normalize_drivetrain(text):
    if not text:
        return ""
    first_token = text.strip().split(" ")[0]
    if first_token in ("FWD", "RWD", "4WD", "2WD", "AWD"):
        return first_token
    if "4 bánh" in text:
        return "4WD"
    if "cầu trước" in text:
        return "FWD"
    if "cầu sau" in text:
        return "RWD"
    return ""


def parse_detail(url):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    row = {k: "" for k in FIELDNAMES}

    brand, model = extract_brand_model(soup)
    row["brand"] = brand
    row["model"] = model
    row["location"] = LOCATION
    row["price_vnd"] = extract_price_vnd(soup, html)

    text_blob = soup.get_text("\n", strip=True)
    raw_specs = {}
    for label in SPEC_LABELS:
        m = re.search(rf"{re.escape(label)}\s*:?\s*\n?([^\n]+)", text_blob)
        if m:
            raw_specs[label] = m.group(1).strip()

    row["year"] = raw_specs.get("Năm sản xuất", "")
    row["condition"] = raw_specs.get("Tình trạng", "")
    row["origin"] = raw_specs.get("Xuất xứ", "")
    row["body_type"] = raw_specs.get("Kiểu dáng", "")
    row["transmission"] = raw_specs.get("Hộp số", "")
    row["engine"] = raw_specs.get("Động cơ", "")
    row["exterior_color"] = raw_specs.get("Màu ngoại thất", "")
    row["seats"] = normalize_seats(raw_specs.get("Số chỗ ngồi", ""))
    row["drivetrain"] = normalize_drivetrain(raw_specs.get("Dẫn động", ""))

    # Mileage extraction handling multiple spec keys or fallback regex
    raw_mileage = raw_specs.get("Số km đã đi") or raw_specs.get("Số km", "")
    if raw_mileage:
        digits_only = re.sub(r"[^\d]", "", raw_mileage)
        row["mileage_km"] = int(digits_only) if digits_only else ""
    else:
        km_match = re.search(r"(?:số\s*km|km\s*đã\s*đi|đã đi)\s*:?\s*([\d.,]+)\s*km", text_blob, re.IGNORECASE)
        if km_match:
            digits_only = re.sub(r"[^\d]", "", km_match.group(1))
            row["mileage_km"] = int(digits_only) if digits_only else ""

    title_tag = soup.select_one("h1")
    title = title_tag.get_text(strip=True) if title_tag else url

    return row, title


def main():
    print("Collecting listing URLs...")
    detail_urls = collect_detail_urls()[:TARGET_ROWS]
    print(f"Found {len(detail_urls)} unique listings. Fetching details...\n")

    rows = []
    for i, url in enumerate(detail_urls, 1):
        try:
            row, title = parse_detail(url)
            rows.append(row)
            print(f"[{i}/{len(detail_urls)}] {title[:60]}")
        except Exception as e:
            print(f"[{i}] failed on {url}: {e}")
        time.sleep(random.uniform(*REQUEST_DELAY))

    # Deduplicate rows based on key fields to ensure no exact duplicates
    seen = set()
    unique_rows = []
    for r in rows:
        identifier = (r["brand"], r["model"], r["year"], r["price_vnd"], r["engine"])
        if identifier not in seen:
            seen.add(identifier)
            unique_rows.append(r)

    out_path = "bonbanh_used_cars.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f"\nSaved {len(unique_rows)} clean rows to {out_path}")


if _name_ == "_main_":
    main()