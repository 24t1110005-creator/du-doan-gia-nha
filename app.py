python -m pip install plotly statsmodels
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
    df['MedHouseVal'] = housing.target
    return df, housing.feature_names

df, feature_names = load_data()
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

st.set_page_config(page_title="Dự đoán Giá Nhà California", layout="wide")
st.title("🏠 Hệ thống Dự đoán Giá nhà")

tab1, tab2 = st.tabs(["📊 Thống kê & Hiệu suất", "🔮 Dự đoán giá"])

with tab1:
    col1, col2 = st.columns(2)
    y_pred = model.predict(X_test)
    
    with col1:
        st.subheader("Biểu đồ Hồi quy (Thực tế vs Dự đoán)")
        fig_reg = px.scatter(x=y_test, y=y_pred, labels={'x': 'Giá thực tế', 'y': 'Giá dự đoán'},
                             opacity=0.5, trendline="ols", trendline_color_override="red")
        st.plotly_chart(fig_reg, use_container_width=True)

    with col2:
        st.subheader("Mức độ quan trọng của đặc trưng")
        feat_df = pd.DataFrame({'Đặc trưng': feature_names, 'Độ quan trọng': model.feature_importances_}).sort_values(by='Độ quan trọng')
        fig_imp = px.bar(feat_df, x='Độ quan trọng', y='Đặc trưng', orientation='h', color='Độ quan trọng')
        st.plotly_chart(fig_imp, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")
    m2.metric("R2 Score", f"{r2_score(y_test, y_pred):.2%}")

with tab2:
    st.subheader("Thông số đầu vào")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        med_inc = st.number_input("Thu nhập TB ($10k)", 0.5, 15.0, 3.5)
        house_age = st.slider("Tuổi nhà", 1, 52, 28)
    with c2:
        ave_rooms = st.slider("Số phòng TB", 1.0, 10.0, 5.0)
        ave_bedrms = st.slider("Phòng ngủ TB", 0.5, 5.0, 1.0)
    with c3:
        pop = st.number_input("Dân số khu vực", 3, 35000, 1400)
        ave_occup = st.number_input("Số người/Hộ", 0.1, 10.0, 3.0)
    
    lat = st.slider("Vĩ độ (Latitude)", 32.0, 42.0, 35.0)
    long = st.slider("Kinh độ (Longitude)", -124.0, -114.0, -119.0)

    if st.button("Dự đoán ngay"):
        user_input = pd.DataFrame([[med_inc, house_age, ave_rooms, ave_bedrms, pop, ave_occup, lat, long]], columns=feature_names)
        res = model.predict(user_input)[0]
        st.success(f"Giá dự đoán: ${res * 100000:,.0f}")
        st.map(pd.DataFrame({'lat': [lat], 'lon': [long]}))
