from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.carvalue_utils import (
    apply_market_guardrail,
    compatibility_subset,
    compatible_values,
    make_user_input,
    market_reference,
)

st.set_page_config(
    page_title="CarValue AI — Used Car Price Estimator",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
DATA_PATH = PROJECT_ROOT / "artifacts" / "CarValue_AI_model_candidate.csv"
META_PATH = PROJECT_ROOT / "models" / "final_model_metadata.json"
CSS_PATH = PROJECT_ROOT / "assets" / "styles.css"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_modeling_data():
    df = pd.read_csv(DATA_PATH)
    # Keep display/compatibility values clean and deterministic.
    for c in ["brand", "model", "origin", "body_type", "transmission", "fuel_type", "exterior_color", "drivetrain"]:
        if c in df.columns:
            df[c] = df[c].replace({"-": np.nan, "nan": np.nan})
    return df


@st.cache_data
def load_metadata():
    if not META_PATH.exists():
        return {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def inject_css():
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def fmt_vnd(value):
    if value is None or not np.isfinite(float(value)):
        return "—"
    v = float(value)
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:,.2f} tỷ"
    return f"{v / 1_000_000:,.0f} triệu"


def option_select(label, values, key, format_func=None, help_text=None):
    values = list(values)
    if not values:
        st.text_input(label, value="Không có dữ liệu / N/A", disabled=True, key=f"{key}_na")
        return None
    kwargs = dict(label=label, options=values, key=key, disabled=(len(values) == 1), help=help_text)
    if format_func is not None:
        kwargs["format_func"] = format_func
    return st.selectbox(**kwargs)


def filter_exact(df, col, value):
    if value is None or col not in df.columns:
        return df
    if pd.api.types.is_numeric_dtype(df[col]):
        return df[pd.to_numeric(df[col], errors="coerce").eq(float(value))]
    return df[df[col].astype(str).eq(str(value))]


if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Thiếu model/artifact. Hãy chạy toàn bộ notebook trước: python run_all_notebooks.py")
    st.stop()

inject_css()
model = load_model()
df = load_modeling_data()
meta = load_metadata()
reference_year = int(meta.get("reference_year", int(pd.to_numeric(df["year"], errors="coerce").max())))

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Data Science · Used-car marketplace</div>
      <div class="hero-title">CarValue AI</div>
      <p class="hero-sub">Ước tính giá đăng bán xe cũ từ các thuộc tính thực tế trong dữ liệu Bonbanh. Form chỉ cho phép các cấu hình đã xuất hiện với đúng dòng xe/năm xe trong dataset, giúp tránh những tổ hợp vô lý khi demo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.12, 0.88], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="section-title">Thông tin xe</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Các lựa chọn kỹ thuật được lọc tuần tự theo cấu hình quan sát được trong dữ liệu.</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        brands = sorted(df["brand"].dropna().astype(str).unique().tolist())
        with c1:
            brand = st.selectbox("Hãng xe", brands, key="brand")
        brand_df = df[df["brand"].astype(str).eq(str(brand))]
        models = sorted(brand_df["model"].dropna().astype(str).unique().tolist())
        with c2:
            model_name = st.selectbox("Dòng xe / phiên bản", models, key="model")

        model_df = compatibility_subset(df, brand, model_name)
        years = sorted(pd.to_numeric(model_df["year"], errors="coerce").dropna().astype(int).unique().tolist(), reverse=True)
        with c1:
            year = st.selectbox("Năm sản xuất", years, key="year", disabled=(len(years) == 1))
        exact = compatibility_subset(df, brand, model_name, int(year))

        # Mileage is the only continuous user-controlled usage variable.
        valid_mileage = pd.to_numeric(exact["mileage_km"], errors="coerce").dropna()
        if len(valid_mileage):
            mileage_default = int(round(float(valid_mileage.median()) / 1000.0) * 1000)
            observed_low = int(valid_mileage.min())
            observed_high = int(valid_mileage.max())
        else:
            mileage_default, observed_low, observed_high = 30000, 0, 200000
        mileage_cap = max(50_000, min(1_000_000, int(max(observed_high * 1.5, mileage_default * 2, 50_000))))
        with c2:
            mileage = st.number_input(
                "Số km đã đi",
                min_value=0,
                max_value=mileage_cap,
                value=min(max(mileage_default, 0), mileage_cap),
                step=1000,
                help=f"Khoảng quan sát cho {brand} {model_name} {year}: {observed_low:,}–{observed_high:,} km" if len(valid_mileage) else None,
            )

        # Sequential filtering guarantees that each next option occurs together
        # with everything already selected above it in at least one listing.
        work = exact.copy()
        with c1:
            origin = option_select("Nguồn gốc", compatible_values(work, "origin"), "origin")
        work = filter_exact(work, "origin", origin)

        with c2:
            body_type = option_select("Kiểu dáng", compatible_values(work, "body_type"), "body_type")
        work = filter_exact(work, "body_type", body_type)

        with c1:
            transmission = option_select("Hộp số", compatible_values(work, "transmission"), "transmission")
        work = filter_exact(work, "transmission", transmission)

        with c2:
            fuel_type = option_select("Nhiên liệu", compatible_values(work, "fuel_type"), "fuel_type")
        work = filter_exact(work, "fuel_type", fuel_type)

        with c1:
            seats_values = compatible_values(work, "seats")
            seats = option_select("Số chỗ", seats_values, "seats", format_func=lambda x: f"{int(float(x))} chỗ")
        work = filter_exact(work, "seats", seats)

        with c2:
            drivetrain = option_select("Dẫn động", compatible_values(work, "drivetrain"), "drivetrain")
        work = filter_exact(work, "drivetrain", drivetrain)

        # Engine displacement is not applicable to EVs. For combustion/hybrid
        # cars, only displacement values actually observed for this compatible
        # configuration are offered.
        is_electric = str(fuel_type).strip().casefold() == "điện".casefold()
        with c1:
            if is_electric:
                st.text_input("Dung tích động cơ", value="N/A — xe điện", disabled=True)
                engine_size = np.nan
            else:
                engine_values = compatible_values(work, "engine_size_l")
                engine_size = option_select(
                    "Dung tích động cơ",
                    engine_values,
                    "engine_size",
                    format_func=lambda x: f"{float(x):.1f} L",
                )
        if not is_electric:
            work = filter_exact(work, "engine_size_l", engine_size)

        with c2:
            exterior_color = option_select("Màu ngoại thất", compatible_values(work, "exterior_color"), "color")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("ƯỚC TÍNH GIÁ", type="primary", use_container_width=True)
        st.markdown(
            '<div class="small-note">* Các trường bị khóa là thuộc tính chỉ có một giá trị tương thích trong dataset. Các lựa chọn phản ánh dữ liệu quan sát được, không phải toàn bộ cấu hình từng được hãng sản xuất.</div>',
            unsafe_allow_html=True,
        )

with right:
    with st.container(border=True):
        st.markdown('<div class="section-title">Kết quả dự đoán</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Final model + kiểm tra tính hợp lý theo các listing cùng dòng xe.</div>', unsafe_allow_html=True)

        if predict_clicked:
            x = make_user_input(
                reference_year=reference_year,
                brand=str(brand),
                model=str(model_name),
                year=int(year),
                mileage_km=float(mileage),
                origin=origin,
                body_type=body_type,
                transmission=transmission,
                fuel_type=fuel_type,
                engine_size_l=engine_size,
                exterior_color=exterior_color,
                seats=seats,
                drivetrain=drivetrain,
            )
            raw_price = float(model.predict(x)[0])
            ref = market_reference(df, str(brand), str(model_name), int(year))
            final_price, guardrail_used = apply_market_guardrail(raw_price, ref)

            model_label = meta.get("selected_model", getattr(model, "name", "Model"))
            guardrail_pill = '<span class="pill pill-warn">Plausibility guardrail applied</span>' if guardrail_used else '<span class="pill">Valid observed configuration</span>'
            st.markdown(
                f"""
                <div class="price-card">
                  <div class="price-label">GIÁ ĐĂNG BÁN ƯỚC TÍNH</div>
                  <div class="price-main">{final_price:,.0f} VND</div>
                  <div class="price-sub">≈ {final_price/1e6:,.1f} triệu VND</div>
                  <div style="margin-top:14px">{guardrail_pill}<span class="pill">{model_label}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            count = int(ref.get("count", 0))
            confidence = "Cao" if count >= 30 else ("Trung bình" if count >= 10 else "Thấp")
            st.markdown(
                f"""
                <div class="stat-grid">
                  <div class="stat-card"><div class="stat-label">Listing tham chiếu</div><div class="stat-value">{count:,}</div></div>
                  <div class="stat-card"><div class="stat-label">Trung vị thị trường</div><div class="stat-value">{fmt_vnd(ref.get('median'))}</div></div>
                  <div class="stat-card"><div class="stat-label">Mức hỗ trợ dữ liệu</div><div class="stat-value">{confidence}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if count:
                st.markdown(
                    f"**Khoảng tham chiếu 50% giữa:** {fmt_vnd(ref.get('q25'))} – {fmt_vnd(ref.get('q75'))} VND  \n"
                    f"Nguồn tham chiếu: **{ref.get('scope', 'model-year')}** trên tập train."
                )

            with st.expander("Chi tiết kỹ thuật & kiểm tra model"):
                st.write(f"Raw model prediction: **{raw_price:,.0f} VND**")
                if guardrail_used:
                    st.warning(
                        "Raw prediction nằm quá xa vùng giá của các listing tương đồng, nên giao diện áp dụng một guardrail rộng để tránh hiển thị giá phi thực tế. Raw prediction vẫn được giữ ở đây để audit."
                    )
                st.dataframe(x, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                """
                <div class="price-card" style="min-height:220px;display:flex;align-items:center;justify-content:center;text-align:center">
                  <div><div class="price-label">CHƯA CÓ DỰ ĐOÁN</div><div style="font-size:18px;font-weight:750;margin-top:8px">Chọn cấu hình xe rồi bấm “Ước tính giá”.</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")
st.markdown(
    "<div class='small-note'><b>Phạm vi:</b> CarValue AI ước tính <b>online asking price</b> trong mẫu Bonbanh, không phải giá giao dịch cuối cùng sau thương lượng. Dữ liệu là một snapshot và không được dùng để suy luận biến động giá theo thời gian.</div>",
    unsafe_allow_html=True,
)
