# California Housing Price Prediction (Machine Learning)

## Objective

Analyze the California Housing dataset to predict median house values using multiple machine learning algorithms and evaluate their performance based on Root Mean Squared Error (RMSE).
The project also explores the key factors influencing house prices across regions.
Machine learning analysis comparing multiple algorithms to predict median house values using the California housing dataset.

## Project Overview

**Business Problem:**
Understanding which features (like income, proximity to the ocean, rooms, etc.) most impact housing prices can help developers, investors, and policymakers make data-driven decisions.

**Goal:**
Build regression models to predict housing prices and compare their accuracy using RMSE to identify the most effective algorithm.

---

## Live Demo

**Try the app now**: https://california-housing-predictor-ml.streamlit.app/

---

## Dataset Information
**Source:** `housing.csv`  
**Features:**
- **Numerical:** `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`
- **Categorical:** `ocean_proximity`
- **Target:** `median_house_value`

**Preprocessing Steps:**
- Converted categorical column `ocean_proximity` → dummy variables  
- Removed null values and shuffled dataset  
- Split data into Train (~18,000), Validation (~1,200), and Test (~1,200)  
- Standardized numerical columns using `StandardScaler`

---

## Exploratory Data Analysis (EDA)
- `median_income` strongly correlates with `median_house_value`
- Population, households, and total rooms show internal correlation
- Houses **near the ocean** have higher prices
- **Latitude vs Longitude** scatterplots show red clusters (high prices) near coasts

---

## Machine Learning Models Used

| Algorithm | Description | Key Insight |
|------------|--------------|-------------|
| **Linear Regression** | Simple baseline model | Interpretable but less accurate |
| **K-Nearest Neighbors (KNN)** | Instance-based model | Sensitive to scaling, good for small data |
| **Random Forest Regressor** | Ensemble of trees | Handles non-linearity well |
| **Gradient Boosting Regressor** | Sequential boosting | **Best traditional model** |
| **Neural Networks (Simple / Medium / Large)** | Deep learning models | Capture complex patterns; may overfit |

---

## Files
- `Predicting_California_Housing_Prices.ipynb` - Main analysis notebook
- `housing.csv` - Dataset
- `models/` - Trained neural network models

---

## Visualizations
- Correlation Heatmap  
- Latitude vs Longitude Price Map  
- Actual vs Predicted Scatter Plots  
- Training vs Validation RMSE  
- Feature Histograms  

---

## Technologies & Libraries
**Language:** Python  
**Libraries:**
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (Linear Regression, KNN, Random Forest, Gradient Boosting)
- `tensorflow.keras` (Neural Networks)
- `statsmodels`, `scipy` (Statistical analysis)

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/california-housing-price-prediction.git
cd california-housing-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Predicting_California_Housing_Prices.ipynb

