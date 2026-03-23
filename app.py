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
st.title("🏠 Hệ thống Phân tích & Dự đoán Giá nhà")

st.sidebar.header("📈 Chỉ số Thị trường 2026")
lai_suat = st.sidebar.slider("Lãi suất vay (%)", 5.0, 15.0, 8.5)
lam_phat = st.sidebar.slider("Tỷ lệ lạm phát (%)", 0.0, 10.0, 4.0)
nhu_
