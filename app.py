import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

@st.cache_data
def load_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Gia_Nha'] = housing.target
    return df, housing.feature_names

df, feature_names = load_data()

ten_tieng_viet = {
    'MedInc': 'Thu nhập TB',
    'HouseAge': 'Tuổi nhà',
    'AveRooms': 'Số phòng TB',
    'AveBedrms': 'Phòng ngủ TB',
    'Population': 'Dân số',
    'AveOccup': 'Số người/Hộ',
    'Latitude': 'Vĩ độ',
    'Longitude': 'Kinh độ'
}

X = df.drop('Gia_Nha', axis=1)
y = df['Gia_Nha']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

st.set_page_config(page_title="Chuyên gia Định giá Nhà", layout="wide")
st.title("🏠 Hệ thống Phân tích & Dự đoán Giá nhà")

st.sidebar.header("📈 Chỉ số Thị trường 2026")
lai_suat = st.sidebar.slider("Lãi suất vay (%)", 5.0, 12.0, 8.5)

tab1, tab2 = st.tabs(["📊 Thống kê & Hiệu suất", "🚀 Dự đoán giá"])

with tab1:
    col1, col2 = st.columns(2)
    y_pred = model.predict(X_test)
    
    with col1:
        st.subheader("Biểu đồ Hồi quy (Thực tế vs Dự đoán)")
        fig_reg = px.scatter(x=y_test, y=y_pred, 
                             labels={'x': 'Giá thực tế ($100k)', 'y': 'Giá dự đoán ($100k)'},
                             opacity=0.5, trendline="ols", trendline_color_override="red")
        st.plotly_chart(fig_reg, use_container_width=True)

    with col2:
        st.subheader("Mức độ quan trọng của đặc trưng")
        vi_features = [ten_tieng_viet.get(f, f) for f in feature_names]
        feat_df = pd.DataFrame({'Đặc trưng': vi_features, 'Độ quan trọng': model.feature_importances_}).sort_values(by='Độ quan trọng')
        fig_imp = px.bar(feat_df, x='Độ quan trọng', y='Đặc trưng', orientation='h', color='Độ quan trọng')
        st.plotly_chart(fig_imp, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Sai số MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")
    m2.metric("Độ chính xác R2", f"{r2_score(y_test, y_pred):.2%}")

with tab2:
    st.subheader("Nhập thông số bất động sản")
    
    st.markdown("#### 📍 Vị trí (Location)")
    khu_vuc = st.selectbox("Chọn khu vực", 
                           ["San Francisco (Vùng Bắc)", "Los Angeles (Vùng Nam)", 
                            "San Diego (Cận Nam)", "Sacramento (Nội địa)", "Tùy chỉnh tọa độ"])
    
    toa_do_vung = {
        "San Francisco (Vùng Bắc)": (37.77, -122.41),
        "Los Angeles (Vùng Nam)": (34.05, -118.24),
        "San Diego (Cận Nam)": (32.71, -117.16),
        "Sacramento (Nội địa)": (38.58, -121.49),
        "Tùy chỉnh tọa độ": (35.0, -119.0)
    }
    lat_def, long_def = toa_do_vung[khu_vuc]

    c1, c2, c3 = st.columns(3)
    with c1:
        med_inc = st.number_input("Thu nhập TB ($10k)", 0.5, 15.0, 3.5)
        house_age = st.slider("Tuổi thọ nhà (năm)", 1, 52, 28)
    with c2:
        ave_rooms = st.slider("Số phòng TB", 1.0, 10.0, 5.0)
        ave_bedrms = st.slider("Phòng ngủ TB", 0.5, 5.0, 1.0)
    with c3:
        pop = st.number_input("Dân số khu vực", 3, 35000, 1400)
        ave_occup = st.number_input("Người ở TB/Hộ", 0.1, 10.0, 3.0)
    
    if khu_vuc == "Tùy chỉnh tọa độ":
        lat = st.slider("Vĩ độ (Lat)", 32.0, 42.0, lat_def)
        long = st.slider("Kinh độ (Long)", -124.0, -114.0, long_def)
    else:
        lat, long = lat_def, long_def
        st.info(f"Đã khóa tọa độ chuẩn cho {khu_vuc}")

    if st.button("🚀 Bắt đầu định giá"):
        user_input = pd.DataFrame([[med_inc, house_age, ave_rooms, ave_bedrms, pop, ave_occup, lat, long]], 
                                  columns=feature_names)
        res = model.predict(user_input)[0]
        gia_cu
