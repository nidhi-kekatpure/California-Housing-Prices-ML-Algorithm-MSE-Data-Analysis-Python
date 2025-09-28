import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 California Housing Price Predictor")
st.markdown("Predict median house values using various machine learning algorithms")

# Load and prepare data
@st.cache_data
def load_data():
    """Load and preprocess the housing data"""
    housing_pd = pd.read_csv('housing.csv')
    housing_pd_shuffled = housing_pd.sample(n=len(housing_pd), random_state=1)
    
    # One-hot encode ocean_proximity
    housing_pd_final = pd.concat([
        housing_pd_shuffled.drop('ocean_proximity', axis=1),
        pd.get_dummies(housing_pd_shuffled['ocean_proximity'])
    ], axis=1)
    
    return housing_pd_final

# Prepare models
@st.cache_resource
def prepare_models():
    """Train and return all models"""
    housing_pd_final = load_data()
    
    # Split data
    train_pd = housing_pd_final[:18000]
    val_pd = housing_pd_final[19215:]
    
    X_train, y_train = train_pd.to_numpy()[:, :-1], train_pd.to_numpy()[:, -1]
    X_val, y_val = val_pd.to_numpy()[:, :-1], val_pd.to_numpy()[:, -1]
    
    # Preprocessing
    scaler = StandardScaler().fit(X_train[:, :8])
    
    def preprocessor(X):
        A = X.copy()
        A[:, :8] = scaler.transform(A[:, :8])
        return A
    
    X_train_scaled = preprocessor(X_train)
    X_val_scaled = preprocessor(X_val)
    
    # Train models
    models = {}
    
    # Linear Regression
    lm = LinearRegression().fit(X_train_scaled, y_train)
    models['Linear Regression'] = {
        'model': lm,
        'train_rmse': np.sqrt(mean_squared_error(lm.predict(X_train_scaled), y_train)),
        'val_rmse': np.sqrt(mean_squared_error(lm.predict(X_val_scaled), y_val))
    }
    
    # KNN
    knn = KNeighborsRegressor(n_neighbors=10).fit(X_train_scaled, y_train)
    models['K-Nearest Neighbors'] = {
        'model': knn,
        'train_rmse': np.sqrt(mean_squared_error(knn.predict(X_train_scaled), y_train)),
        'val_rmse': np.sqrt(mean_squared_error(knn.predict(X_val_scaled), y_val))
    }
    
    # Random Forest
    rfr = RandomForestRegressor(max_depth=10, random_state=42).fit(X_train_scaled, y_train)
    models['Random Forest'] = {
        'model': rfr,
        'train_rmse': np.sqrt(mean_squared_error(rfr.predict(X_train_scaled), y_train)),
        'val_rmse': np.sqrt(mean_squared_error(rfr.predict(X_val_scaled), y_val))
    }
    
    # Gradient Boosting
    gbr = GradientBoostingRegressor(n_estimators=250, random_state=42).fit(X_train_scaled, y_train)
    models['Gradient Boosting'] = {
        'model': gbr,
        'train_rmse': np.sqrt(mean_squared_error(gbr.predict(X_train_scaled), y_train)),
        'val_rmse': np.sqrt(mean_squared_error(gbr.predict(X_val_scaled), y_val))
    }
    
    return models, scaler, X_val_scaled, y_val

# Load models and data
models, scaler, X_val, y_val = prepare_models()
housing_data = load_data()

# Sidebar for model selection
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a model:",
    list(models.keys())
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Make a Prediction")
    
    # Input features
    col_a, col_b = st.columns(2)
    
    with col_a:
        longitude = st.slider("Longitude", -124.0, -114.0, -118.0, 0.01)
        latitude = st.slider("Latitude", 32.0, 42.0, 37.0, 0.01)
        housing_median_age = st.slider("Housing Median Age", 1.0, 52.0, 25.0, 1.0)
        total_rooms = st.slider("Total Rooms", 100, 10000, 2500, 100)
        total_bedrooms = st.slider("Total Bedrooms", 50, 2000, 500, 50)
    
    with col_b:
        population = st.slider("Population", 100, 5000, 1500, 100)
        households = st.slider("Households", 50, 2000, 500, 50)
        median_income = st.slider("Median Income", 0.5, 15.0, 5.0, 0.1)
        
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
        )
    
    # Create input array
    input_data = np.array([
        longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
        population, households, median_income
    ])
    
    # One-hot encode ocean proximity
    ocean_features = np.zeros(5)
    ocean_mapping = {"<1H OCEAN": 0, "INLAND": 1, "ISLAND": 2, "NEAR BAY": 3, "NEAR OCEAN": 4}
    ocean_features[ocean_mapping[ocean_proximity]] = 1
    
    # Combine features
    full_input = np.concatenate([input_data, ocean_features]).reshape(1, -1)
    
    # Preprocess input
    processed_input = full_input.copy()
    processed_input[:, :8] = scaler.transform(processed_input[:, :8])
    
    # Make prediction
    if st.button("🔮 Predict House Value", type="primary"):
        model = models[selected_model]['model']
        prediction = model.predict(processed_input)[0]
        
        st.success(f"**Predicted House Value: ${prediction:,.2f}**")
        
        # Show model performance
        st.info(f"""
        **{selected_model} Performance:**
        - Training RMSE: ${models[selected_model]['train_rmse']:,.2f}
        - Validation RMSE: ${models[selected_model]['val_rmse']:,.2f}
        """)

with col2:
    st.header("📊 Model Comparison")
    
    # Create performance comparison
    model_names = list(models.keys())
    train_rmse = [models[name]['train_rmse'] for name in model_names]
    val_rmse = [models[name]['val_rmse'] for name in model_names]
    
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Training RMSE': train_rmse,
        'Validation RMSE': val_rmse
    })
    
    # Bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Training RMSE', x=comparison_df['Model'], y=comparison_df['Training RMSE']))
    fig.add_trace(go.Bar(name='Validation RMSE', x=comparison_df['Model'], y=comparison_df['Validation RMSE']))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="RMSE ($)",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance table
    st.subheader("Performance Metrics")
    st.dataframe(comparison_df.round(2), use_container_width=True)

# Dataset overview
st.header("📈 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(housing_data))
    
with col2:
    st.metric("Features", housing_data.shape[1] - 1)
    
with col3:
    avg_price = housing_data['median_house_value'].mean()
    st.metric("Average House Value", f"${avg_price:,.2f}")

# Feature distributions
st.subheader("Feature Distributions")
feature_to_plot = st.selectbox(
    "Select feature to visualize:",
    ['median_house_value', 'median_income', 'housing_median_age', 'total_rooms', 'population']
)

fig = px.histogram(housing_data, x=feature_to_plot, nbins=50, title=f"Distribution of {feature_to_plot}")
st.plotly_chart(fig, use_container_width=True)

# Geographic visualization
st.subheader("Geographic Distribution")
fig = px.scatter(
    housing_data, 
    x='longitude', 
    y='latitude', 
    color='median_house_value',
    size='population',
    hover_data=['median_income'],
    title="California Housing Prices by Location",
    color_continuous_scale='viridis'
)
st.plotly_chart(fig, use_container_width=True)
