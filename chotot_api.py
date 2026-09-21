import time
import requests

from utils.data_cleaner import clean_text, clean_number, parse_price_vnd


LISTING_API_URL = "https://gateway.chotot.com/v1/public/ad-listing"
DETAIL_API_URL = "https://gateway.chotot.com/v1/public/ad-detail"

BINH_DUONG_REGION = "10206"
CAR_CATEGORY = "4020"

PAGE_SIZE = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}


OUTPUT_COLUMNS = [
    "Brand",
    "model",
    "year",
    "price_vnd",
    "mileage/km",
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


def request_json(url, params=None, retries=5):
    """
    Send a GET request with retry and exponential backoff.
    """

    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=30,
            )

            print(f"[HTTP] {response.status_code} - {response.url}")

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                wait_time = 2 ** attempt
                print(
                    f"[429] Rate limited. "
                    f"Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
                continue

            if response.status_code >= 500:
                wait_time = 2 ** attempt
                print(
                    f"[{response.status_code}] Server error. "
                    f"Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
                continue

            print(f"[ERROR] Unexpected status: {response.status_code}")
            print(response.text[:1000])
            return None

        except requests.RequestException as error:
            wait_time = 2 ** attempt

            print(
                f"[REQUEST ERROR] {error}. "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

    return None


def test_chotot_api():
    test_cases = [
        {
            "name": "Cars + Binh Duong",
            "params": {
                "cg": "4020",
                "region_v2": "10206",
                "limit": 20,
                "o": 0,
                "w": 1,
                "st": "s,k",
            },
        },
        {
            "name": "Cars only",
            "params": {
                "cg": "4020",
                "limit": 20,
                "o": 0,
                "w": 1,
                "st": "s,k",
            },
        },
        {
            "name": "Cars + Binh Duong without condition",
            "params": {
                "cg": "4020",
                "region_v2": "10206",
                "limit": 20,
                "o": 0,
                "w": 1,
                "st": "s,k",
            },
        },
        {
            "name": "Old category + Binh Duong",
            "params": {
                "cg": "2010",
                "region_v2": "10206",
                "limit": 20,
                "o": 0,
                "w": 1,
                "st": "s,k",
            },
        },
    ]

    for test in test_cases:
        print()
        print("=" * 70)
        print(test["name"])
        print("=" * 70)

        response = requests.get(
            LISTING_API_URL,
            params=test["params"],
            headers=HEADERS,
            timeout=30,
        )

        print("STATUS:", response.status_code)
        print("URL:", response.url)
        print("RESPONSE:", response.text[:2000])

    print("=" * 60)
    print("STEP 1 - COLLECTING CHỢ TỐT LISTING IDs")
    print("=" * 60)

    listing_ids = set()
    offset = 0
    total = None

    while True:
        page_number = (offset // PAGE_SIZE) + 1

        print(
            f"[Listing] Requesting batch {page_number} "
            f"(offset={offset})..."
        )

        params = {
            "cg": CAR_CATEGORY,
            "region_v2": BINH_DUONG_REGION,
            "condition": 2,
            "limit": PAGE_SIZE,
            "o": offset,
            "w": 1,
            "st": "s,k",
        }

        data = request_json(LISTING_API_URL, params=params)

        if not data:
            print("[STOP] Empty API response.")
            break

        ads = data.get("ads", [])

        if total is None:
            total = data.get("total")

            print(f"[API] Reported total: {total}")

        if not ads:
            print(
                f"[STOP] No ads returned at offset {offset}."
            )
            break

        batch_new = 0

        for ad in ads:
            listing_id = (
                ad.get("list_id")
                or ad.get("listId")
                or ad.get("id")
            )

            if listing_id is None:
                continue

            listing_id = str(listing_id)

            if listing_id not in listing_ids:
                listing_ids.add(listing_id)
                batch_new += 1

        print(
            f"[Listing] Received: {len(ads)} | "
            f"New unique IDs: {batch_new} | "
            f"Total unique: {len(listing_ids)}"
        )

        if total is not None and len(listing_ids) >= total:
            print("[STOP] Reached API reported total.")
            break

        if len(ads) < PAGE_SIZE:
            print(
                "[STOP] Last batch contains fewer than "
                f"{PAGE_SIZE} ads."
            )
            break

        offset += PAGE_SIZE

        # Small delay to reduce request pressure.
        time.sleep(0.7)

    print()
    print(f"TOTAL UNIQUE LISTINGS: {len(listing_ids)}")

    return list(listing_ids)


def get_listing_details(listing_id):
    """
    Retrieve detailed information for one listing.
    """

    url = f"{DETAIL_API_URL}/{listing_id}"

    data = request_json(url)

    if not data:
        return None

    # Some API responses wrap the actual listing.
    if isinstance(data.get("ad"), dict):
        return data["ad"]

    if isinstance(data.get("data"), dict):
        return data["data"]

    return data


def get_value(data, *keys):
    """
    Return the first non-empty value from multiple possible keys.
    """

    for key in keys:
        value = data.get(key)

        if value is not None and value != "":
            return value

    return None


def normalize_transmission(value):
    """
    Normalize Chợ Tốt gearbox values.
    """

    if value is None:
        return None

    mapping = {
        1: "Automatic",
        2: "Manual",
        3: "Semi-automatic",
    }

    try:
        numeric_value = int(value)

        if numeric_value in mapping:
            return mapping[numeric_value]

    except (TypeError, ValueError):
        pass

    text = clean_text(value).lower()

    if "tự động" in text or "automatic" in text:
        return "Automatic"

    if "số tay" in text or "manual" in text:
        return "Manual"

    if "bán tự động" in text or "semi" in text:
        return "Semi-automatic"

    return clean_text(value)


def normalize_record(data):
    """
    Convert a raw Chợ Tốt listing into the required schema.
    """

    brand = get_value(
        data,
        "carbrand_name",
        "carbrand",
        "brand",
    )

    model = get_value(
        data,
        "carmodel_name",
        "carmodel",
        "model",
    )

    year = get_value(
        data,
        "manufacture_date",
        "year",
    )

    price = get_value(
        data,
        "price",
    )

    mileage = get_value(
        data,
        "kilometers",
        "mileage",
    )

    condition = get_value(
        data,
        "condition_ad",
        "condition",
    )

    origin = get_value(
        data,
        "origin_name",
        "origin",
    )

    body_type = get_value(
        data,
        "carbody_name",
        "carbody",
        "body_type",
    )

    transmission = get_value(
        data,
        "gearbox",
        "transmission",
    )

    fuel = get_value(
        data,
        "fuel",
    )

    engine_capacity = get_value(
        data,
        "engine_capacity",
    )

    color = get_value(
        data,
        "carcolor_name",
        "carcolor",
        "exterior_color",
    )

    seats = get_value(
        data,
        "seats",
    )

    drivetrain = get_value(
        data,
        "drivetrain",
    )

    location = get_value(
        data,
        "area_name",
        "region_name",
        "location",
    )

    # Build engine field.
    engine_parts = []

    if fuel:
        engine_parts.append(clean_text(fuel))

    if engine_capacity:
        engine_parts.append(clean_text(engine_capacity))

    engine = " | ".join(engine_parts) if engine_parts else None

    # Normalize condition.
    if condition is not None:
        condition_text = clean_text(condition)

        if str(condition).lower() in {"2", "used"}:
            condition = "Used"
        elif condition_text:
            condition = condition_text
    else:
        condition = "Used"

    # Fallback location.
    if not location:
        location = "Binh Duong"

    return {
        "Brand": clean_text(brand),
        "model": clean_text(model),
        "year": clean_number(year),
        "price_vnd": parse_price_vnd(price),
        "mileage/km": clean_number(mileage),
        "condition": condition,
        "origin": clean_text(origin),
        "body_type": clean_text(body_type),
        "transmission": normalize_transmission(transmission),
        "engine": engine,
        "exterior_color": clean_text(color),
        "seats": clean_number(seats),
        "drivetrain": clean_text(drivetrain),
        "location": clean_text(location),
    }


def scrape_chotot_binh_duong():
    """
    Main Chợ Tốt Binh Duong scraper.
    """

    listing_ids = collect_listing_ids()

    if not listing_ids:
        print("[ERROR] No listings were collected.")
        return []

    print()
    print("=" * 60)
    print("STEP 2 - COLLECTING LISTING DETAILS")
    print("=" * 60)

    records = []

    detail_success = 0
    detail_failed = 0

    for index, listing_id in enumerate(listing_ids, start=1):

        print(
            f"[Detail] {index}/{len(listing_ids)} "
            f"-> ID {listing_id}"
        )

        detail = get_listing_details(listing_id)

        if detail:
            record = normalize_record(detail)

            records.append(record)
            detail_success += 1

        else:
            detail_failed += 1

        # Small delay between detail requests.
        time.sleep(0.5)

    print()
    print("=" * 60)
    print("SCRAPING SUMMARY")
    print("=" * 60)

    print(f"Listing IDs collected : {len(listing_ids)}")
    print(f"Detail success        : {detail_success}")
    print(f"Detail failed         : {detail_failed}")
    print(f"Final records         : {len(records)}")

    return records

if __name__ == "__main__":
    test_chotot_api()