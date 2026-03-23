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
    'MedInc': 'Thu nhập TB', 'HouseAge': 'Tuổi nhà', 'AveRooms': 'Số phòng TB',
    'AveBedrms': 'Phòng ngủ TB', 'Population': 'Dân số', 'AveOccup': 'Số người/Hộ',
    'Latitude': 'Vĩ độ', 'Longitude': 'Kinh độ'
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
st.title("🏠 Hệ thống Phân tích & Dự đoán Giá nhà ở CALI")

st.sidebar.header("📈 Chỉ số Thị trường 2026")
lai_suat = st.sidebar.slider("Lãi suất vay (%)", 5.0, 15.0, 8.5)
lam_phat = st.sidebar.slider("Tỷ lệ lạm phát (%)", 0.0, 10.0, 4.0)
nhu_cau = st.sidebar.select_slider("Nhu cầu thị trường", options=[0.8, 0.9, 1.0, 1.1, 1.2], value=1.0)
phi_xay_dung = st.sidebar.slider("Biến động chi phí xây dựng (%)", -10, 20, 0)
chi_so_vung = st.sidebar.slider("Chỉ số kinh tế vùng (Hệ số)", 0.9, 1.5, 1.0)

he_so_thi_truong = nhu_cau * (1 + (lam_phat / 100)) * (1 + (phi_xay_dung / 100)) * chi_so_vung

tab1, tab2 = st.tabs(["📊 Thống kê & Hiệu suất", "🚀 Dự đoán giá"])

with tab1:
    y_pred_goc = model.predict(X_test)
    y_pred_moi = y_pred_goc * he_so_thi_truong
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Hệ số thị trường", f"x{he_so_thi_truong:.2f}")
    col_m2.metric("Giá dự đoán TB", f"${y_pred_moi.mean() * 100000:,.0f}")
    col_m3.metric("R2 Score (Gốc)", f"{r2_score(y_test, y_pred_goc):.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Biểu đồ Hồi quy (Cập nhật theo chỉ số)")
        fig_reg = px.scatter(x=y_test, y=y_pred_moi, 
                             labels={'x': 'Giá thực tế ($100k)', 'y': 'Giá dự đoán ĐÃ ĐIỀU CHỈNH ($100k)'},
                             opacity=0.5, color_discrete_sequence=['#3366FF'])
        fig_reg.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                          line=dict(color="Red", dash="dot"))
        st.plotly_chart(fig_reg, use_container_width=True)

    with col2:
        st.write("### Mức độ quan trọng của đặc trưng")
        vi_features = [ten_tieng_viet.get(f, f) for f in feature_names]
        feat_df = pd.DataFrame({'Đặc trưng': vi_features, 'Độ quan trọng': model.feature_importances_}).sort_values(by='Độ quan trọng')
        fig_imp = px.bar(feat_df, x='Độ quan trọng', y='Đặc trưng', orientation='h', color='Độ quan trọng')
        st.plotly_chart(fig_imp, use_container_width=True)

with tab2:
    st.subheader("Nhập thông số bất động sản")
    
    st.markdown("#### 📍 Vị trí (Location)")
    khu_vuc = st.selectbox("Chọn khu vực", 
                           ["San Francisco (Vùng Bắc)", "Los Angeles (Vùng Nam)", 
                            "San Diego (Cận Nam)", "Sacramento (Nội địa)", "Tùy chỉnh tọa độ"])
    
    toa_do_vung = {
        "San Francisco (Vùng Bắc)": (37.77, -122.41), "Los Angeles (Vùng Nam)": (34.05, -118.24),
        "San Diego (Cận Nam)": (32.71, -117.16), "Sacramento (Nội địa)": (38.58, -121.49),
        "Tùy chỉnh tọa độ": (35.0, -119.0)
    }
    lat_def, long_def = toa_do_vung[khu_vuc]

    c1, c2, c3 = st.columns(3)
    with c1:
        med_inc = st.number_input("Thu nhập TB ($10k)", 0.5, 15.0, 3.5)
        house_age = st.slider("Tuổi thọ nhà", 1, 52, 28)
    with c2:
        ave_rooms = st.slider("Số phòng (Trung bình)", 1.0, 15.0, 5.0)
        ave_bedrms = st.slider("Phòng ngủ (Trung bình)", 0.5, 5.0, 1.0)
    with c3:
        pop = st.number_input("Dân số khu vực", 3, 35000, 1400)
        ave_occup = st.number_input("Người ở trung bình/Hộ", 0.1, 10.0, 3.0)
    
    if khu_vuc == "Tùy chỉnh tọa độ":
        lat = st.slider("Vĩ độ", 32.0, 42.0, lat_def)
        long = st.slider("Kinh độ", -124.0, -114.0, long_def)
    else:
        lat, long = lat_def, long_def

    if st.button("🚀 Bắt đầu định giá"):
        user_input = pd.DataFrame([[med_inc, house_age, ave_rooms, ave_bedrms, pop, ave_occup, lat, long]], 
                                  columns=feature_names)
        res_raw = model.predict(user_input)[0]
        res_final = res_raw * he_so_thi_truong
        gia_that = res_final * 100000
        
        res_c1, res_c2 = st.columns(2)
        with res_c1:
            st.success(f"### Giá dự đoán: ${gia_that:,.0f}")
            thanh_toan = (gia_that * (1 + lai_suat/100)) / 240
            st.metric("Trả góp ước tính (240 tháng)", f"${thanh_toan:,.2f}/tháng")
        
        with res_c2:
            st.map(pd.DataFrame({'lat': [lat], 'lon': [long]}))
        st.balloons()
