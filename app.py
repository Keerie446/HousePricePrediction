"""
Streamlit app for House Price Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegant fonts
st.markdown("""
    <style>
    h1 {
        font-family: 'Times New Roman', Times, serif;
        font-size: 2.5em;
        font-weight: normal;
        letter-spacing: 1px;
    }
    h2 {
        font-family: 'Times New Roman', Times, serif;
        font-weight: normal;
        letter-spacing: 0.5px;
    }
    h3 {
        font-family: 'Times New Roman', Times, serif;
        font-weight: normal;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    model_path = Path("models/model.joblib")
    if not model_path.exists():
        st.error("Model not found. Please train the model first.")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_data():
    return pd.read_csv("data/housing.csv")

# Title and description
st.title("House Price Predictor")
st.markdown("""
This app predicts house prices based on features like area, bedrooms, bathrooms, and amenities.
Enter your house details in the sidebar and get an instant price estimate!
""")

# Load resources
pipeline = load_model()
df_train = load_data()

# Sidebar for input
st.sidebar.header("House Details")

# Create input form
with st.sidebar.form("prediction_form"):
    st.subheader("Enter House Features")
    
    area = st.slider("Area (sq ft)", min_value=int(df_train['area'].min()), 
                     max_value=int(df_train['area'].max()), 
                     value=6000, step=100)
    
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=6, value=3, step=1)
    
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=4, value=2, step=1)
    
    stories = st.slider("Stories", min_value=1, max_value=4, value=2, step=1)
    
    parking = st.slider("Parking Spaces", min_value=0, max_value=3, value=1, step=1)
    
    mainroad = st.selectbox("Main Road Access", ["yes", "no"])
    
    guestroom = st.selectbox("Guest Room", ["yes", "no"])
    
    basement = st.selectbox("Basement", ["yes", "no"])
    
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
    
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
    
    prefarea = st.selectbox("Preferred Area", ["yes", "no"])
    
    furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])
    
    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# Make prediction
if submitted:
    # Create input dataframe
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })
    
    # Predict
    prediction = pipeline.predict(input_data)[0]
    
    # Display result
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.success("Prediction Complete!")
        st.metric(label="Estimated Price", value=f"₹{prediction:,.0f}")
    
    with col2:
        # Price range context
        min_price = df_train['price'].min()
        max_price = df_train['price'].max()
        mean_price = df_train['price'].mean()
        
        st.info(f"""
        **Training Data Context:**
        - Min Price: ₹{min_price:,.0f}
        - Mean Price: ₹{mean_price:,.0f}
        - Max Price: ₹{max_price:,.0f}
        """)

# Tabs for visualizations
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["Data Overview", "Price Distribution", "Feature Analysis"])

with tab1:
    st.subheader("Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(df_train))
    with col2:
        st.metric("Avg Price", f"₹{df_train['price'].mean():,.0f}")
    with col3:
        st.metric("Avg Area", f"{df_train['area'].mean():,.0f} sq ft")
    with col4:
        st.metric("Avg Bedrooms", f"{df_train['bedrooms'].mean():.1f}")
    
    st.dataframe(df_train.describe(), use_container_width=True)

with tab2:
    # Price distribution
    fig_price = px.histogram(df_train, x='price', nbins=30, 
                             title="Price Distribution",
                             labels={'price': 'Price (₹)'})
    fig_price.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Price vs Area
    fig_scatter = px.scatter(df_train, x='area', y='price', 
                            title="Price vs Area",
                            labels={'area': 'Area (sq ft)', 'price': 'Price (₹)'},
                            opacity=0.6)
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    # Feature impact visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Bedrooms vs Price
        fig_bed = px.box(df_train, x='bedrooms', y='price',
                        title="Price by Number of Bedrooms")
        fig_bed.update_layout(height=400)
        st.plotly_chart(fig_bed, use_container_width=True)
    
    with col2:
        # AC impact
        fig_ac = px.box(df_train, x='airconditioning', y='price',
                       title="Price Impact: Air Conditioning")
        fig_ac.update_layout(height=400)
        st.plotly_chart(fig_ac, use_container_width=True)
    
    # Furnishing status
    fig_furn = px.box(df_train, x='furnishingstatus', y='price',
                     title="Price by Furnishing Status")
    fig_furn.update_layout(height=400)
    st.plotly_chart(fig_furn, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>House Price Predictor | Week 1 Internship Project</p>
    <p style='font-size: 0.8em; color: #666;'>
    Built with Streamlit | Model: Random Forest Regressor
    </p>
</div>
""", unsafe_allow_html=True)
