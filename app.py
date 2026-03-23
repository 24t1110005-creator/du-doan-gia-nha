import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Dự đoán Giá Nhà", layout="wide")

data = {
    'dien_tich': [50, 80, 120, 45, 150, 100, 75, 200, 60, 110, 85, 130, 55, 180, 95],
    'so_phong_ngu': [1, 2, 3, 1, 4, 3, 2, 5, 2, 3, 2, 3, 1, 4, 3],
    'quan': ['Quận 1', 'Quận 3', 'Quận 7', 'Quận 1', 'Quận 7', 'Quận 3', 'Quận 1', 'Quận 2', 'Quận 3', 'Quận 2', 'Quận 1', 'Quận 7', 'Quận 3', 'Quận 2', 'Quận 7'],
    'gia_nha': [2.5, 4.2, 6.0, 2.1, 8.5, 5.5, 3.8, 12.0, 3.2, 7.0, 4.0, 6.5, 2.8, 10.5, 5.2]
}
df = pd.DataFrame(data)

X = df.drop('gia_nha', axis=1)
y = df['gia_nha']

numeric_features = ['dien_tich', 'so_phong_ngu']
categorical_features = ['quan']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

base_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 5, 10],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_house_model.pkl')

y_pred = best_model.predict(X_test)
y_test_flat = np.ravel(y_test)
y_pred_flat = np.ravel(y_pred)

st.title("🏠 Ứng dụng Dự đoán Giá Nhà")
st.divider()

tab1, tab2 = st.tabs(["🔍 Phân tích mô hình", "🚀 Dự đoán giá"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Biểu đồ Hồi quy")
        fig1, ax1 = plt.subplots()
        sns.regplot(x=y_test_flat, y=y_pred_flat, ax=ax1, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        ax1.set_xlabel("Giá thực tế")
        ax1.set_ylabel("Giá dự đoán")
        st.pyplot(fig1)

    with col2:
        st.write("### Độ quan trọng đặc trưng")
        importances = best_model.named_steps['regressor'].feature_importances_
        ohe_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(['quan'])
        feature_names = numeric_features + list(ohe_names)
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        fig2, ax2 = plt.subplots()
        feat_imp.plot(kind='bar', ax=ax2)
        st.pyplot(fig2)

    st.write("### So sánh giá thực tế vs Dự đoán")
    st.bar_chart(pd.DataFrame({'Thực tế': y_test_flat, 'Dự đoán': y_pred_flat}))

with tab2:
    st.subheader("Nhập thông số bất động sản")
    c1, c2, c3 = st.columns(3)
    
    dt_val = c1.number_input("Diện tích (m2)", 20, 500, 85)
    sp_val = c2.slider("Số phòng ngủ", 1, 5, 2)
    q_val = c3.selectbox("Chọn Quận", df['quan'].unique())
    
    if st.button("Dự đoán"):
        input_df = pd.DataFrame({'dien_tich': [dt_val], 'so_phong_ngu': [sp_val], 'quan': [q_val]})
        res = best_model.predict(input_df)[0]
        st.metric("Giá dự đoán", f"{res:.2f} Tỷ VNĐ")
        st.balloons()

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")