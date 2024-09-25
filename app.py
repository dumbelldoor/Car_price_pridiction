import pandas as pd
import streamlit as st
import pickle as pkl
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import LinearRegression, Lasso, Ridge
# from sklearn.metrics import mean_squared_error, r2_score


# Load dataset (for feature range reference)
df = pd.read_csv('cardekho_dataset.csv')

# Load the best model (previously saved)
with open('best_model_RandomForest Regressor.pkl', 'rb') as file:
    best_model = pkl.load(file)

# Load the scaler used during training
with open('Scaler.pkl', 'rb') as file:
    scaler = pkl.load(file)

# Sidebar for user input
st.sidebar.title('Car Price Predictor')
st.sidebar.write('Input the parameters to predict car price')

# Function to get user input from sidebar
def user_input_features():
    brand = st.sidebar.selectbox('Brand', df['brand'].unique())
    vehicle_age = st.sidebar.slider('Vehicle Age (Years)', int(df['vehicle_age'].min()), int(df['vehicle_age'].max()), int(df['vehicle_age'].mean()))
    km_driven = st.sidebar.slider('KM Driven', int(df['km_driven'].min()), int(df['km_driven'].max()), int(df['km_driven'].mean()))
    fuel_type = st.sidebar.selectbox('Fuel Type', df['fuel_type'].unique())
    seller_type = st.sidebar.selectbox('Seller Type', df['seller_type'].unique())
    engine = st.sidebar.slider('Engine (cc)', float(df['engine'].min()), float(df['engine'].max()), float(df['engine'].mean()))
    transmission_type = st.sidebar.selectbox('Transmission Type', df['transmission_type'].unique())
    mileage = st.sidebar.slider('Mileage (km/l)', float(df['mileage'].min()), float(df['mileage'].max()), float(df['mileage'].mean()))
    max_power = st.sidebar.slider('Max Power (bhp)', float(df['max_power'].min()), float(df['max_power'].max()), float(df['max_power'].mean()))
    seats = st.sidebar.slider('Seats', int(df['seats'].min()), int(df['seats'].max()), int(df['seats'].mean()))

    # Return as dictionary
    return {
        'brand': brand,
        'vehicle_age': vehicle_age,
        'km_driven': km_driven,
        'fuel_type': fuel_type,
        'seller_type': seller_type,
        'engine': engine,
        'transmission_type': transmission_type,
        'mileage': mileage,
        'max_power': max_power,
        'seats': seats
    }

# Get user input
input_data = user_input_features()

# Encode categorical features using .astype('category').cat.codes
input_data['brand'] = df['brand'].astype('category').cat.codes[df['brand'] == input_data['brand']].values[0]
input_data['fuel_type'] = df['fuel_type'].astype('category').cat.codes[df['fuel_type'] == input_data['fuel_type']].values[0]
input_data['seller_type'] = df['seller_type'].astype('category').cat.codes[df['seller_type'] == input_data['seller_type']].values[0]
input_data['transmission_type'] = df['transmission_type'].astype('category').cat.codes[df['transmission_type'] == input_data['transmission_type']].values[0]

# Convert input data into a DataFrame
input_df = pd.DataFrame([input_data])

# Ensure the feature order matches the training data
input_df = input_df[['brand', 'vehicle_age', 'km_driven', 'seller_type','fuel_type',  'transmission_type','mileage', 'engine' , 'max_power', 'seats']]

# Standardize numeric features using the loaded scaler
numeric_cols = ['brand','vehicle_age', 'km_driven','seller_type','fuel_type','transmission_type', 'mileage', 'engine', 'max_power', 'seats']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Prediction
prediction = best_model.predict(input_df)

# Display results
st.header('Car Price Prediction')
st.write(f"The predicted selling price is: **₹{round(prediction[0], 2)}**")

# Add a plot to show key statistics
st.write("Top 5 most frequent brands in the dataset:")
brand_counts = df['brand'].value_counts().head(5)
st.bar_chart(brand_counts)
