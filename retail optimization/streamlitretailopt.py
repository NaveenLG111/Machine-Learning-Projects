import streamlit as st
import pandas as pd
import joblib

# 1. Title and Description
st.title("🛍️ Retail Price Prediction App")
st.write("Enter the product metrics to predict the target unit price.")

# 2. Load the pre-trained, optimized regression pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("best_retail_model.pkl")

try:
    model = load_pipeline()
    
    # 3. Interactive User Input Sidebars / Numerical Fields
    product_category = st.selectbox("Product Category", ["bed_bath_table", "computers_accessories", "health_beauty"])
    qty = st.number_input("Quantity", min_value=1, value=1)
    total_price = st.number_input("Total Transaction Price ($)", min_value=0.0, value=50.0)
    freight_price = st.number_input("Freight Price ($)", min_value=0.0, value=15.0)
    product_score = st.slider("Product Review Score", 1.0, 5.0, 4.0)
    
    # 4. Predict on Button Click
    if st.button("Predict Unit Price"):
        input_data = pd.DataFrame([{
            'product_category_name': product_category, 'qty': qty,
            'total_price': total_price, 'freight_price': freight_price,
            'product_score': product_score, 'product_name_lenght': 40,
            'product_description_lenght': 150, 'product_photos_qty': 2,
            'product_weight_g': 500, 'customers': 50, 'weekday': 22,
            'weekend': 8, 'holiday': 1, 'month': 7, 'year': 2026,
            's': 10.0, 'volume': 4000, 'comp_1': total_price, 'ps1': 4.0,
            'fp1': freight_price, 'comp_2': total_price, 'ps2': 4.0, 'fp2': freight_price,
            'comp_3': total_price, 'ps3': 4.0, 'fp3': freight_price, 'lag_price': total_price
        }])
        
        prediction = model.predict(input_data)
        st.success(f"🎯 Estimated Unit Price: ${prediction[0]:.2f}")

except FileNotFoundError:
    st.error("Model file 'best_retail_model.pkl' not found. Please export your trained pipeline first.")