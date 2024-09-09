import streamlit as st
import pandas as pd
import pickle

# Load the trained model and dataset
model = pickle.load(open('model.pkl', 'rb'))

# Load the dataset for monitoring
df = pd.read_csv('SmartCrop-Dataset.csv')

# Sidebar for navigation

st.sidebar.image("aquacrops_logo.png", width=150)  # Replace with the path to your image file
st.sidebar.title("AquaCrops:")
st.sidebar.markdown("""
**Welcome to AquaCrops!**  
Optimize your crop management with real-time recommendations and monitoring. Choose an option from the sidebar to get started.
""")
option = st.sidebar.selectbox("Select Option", ["Crop Recommendation", "Monitorization"])

# Function to check which parameters need adjustment for a specific crop
def check_parameters(parameters, df, crop_label):
    # Filter the dataset for the specific crop
    crop_data = df[df['label'] == crop_label]

    # Check if there is enough data for the selected crop
    if crop_data.empty:
        return {"error": f"No data available for crop: {crop_label}"}

    # Calculate ideal ranges based on the 5th and 95th percentiles for the specific crop
    ideal_ranges = {
        'Nitrogen': (crop_data['N'].quantile(0.05), crop_data['N'].quantile(0.95)),
        'Phosphorus': (crop_data['P'].quantile(0.05), crop_data['P'].quantile(0.95)),
        'Potassium': (crop_data['K'].quantile(0.05), crop_data['K'].quantile(0.95)),
        'Temperature': (crop_data['temperature'].quantile(0.05), crop_data['temperature'].quantile(0.95)),
        'Humidity': (crop_data['humidity'].quantile(0.05), crop_data['humidity'].quantile(0.95)),
        'pH': (crop_data['ph'].quantile(0.05), crop_data['ph'].quantile(0.95))
    }

    # Compare the given parameters against the ideal ranges for the specific crop
    adjustments = {}
    for param, value in parameters.items():
        if value < ideal_ranges[param][0]:
            adjustments[param] = f"Increase {param} (Current: {value}, Ideal Range: {ideal_ranges[param]})"
        elif value > ideal_ranges[param][1]:
            adjustments[param] = f"Decrease {param} (Current: {value}, Ideal Range: {ideal_ranges[param]})"
        else:
            adjustments[param] = f"{param} is optimal (Current: {value}, Ideal Range: {ideal_ranges[param]})"

    return adjustments

if option == "Crop Recommendation":
    st.title("Crop Recommendation")

    # Input parameters for crop recommendation
    nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, step=0.1)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0, step=0.1)
    potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, step=0.1)
    temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, step=1)
    pH = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

    # Prepare input for prediction with correct feature names
    input_data = pd.DataFrame({
        'N': [nitrogen],
        'P': [phosphorus],
        'K': [potassium],
        'temperature': [temp],
        'humidity': [humidity],
        'ph': [pH],
        'rainfall': [rainfall]
    })

    # Predict and display recommendation
    if st.button("Recommend Crop"):
        prediction = model.predict(input_data)
        st.write(f"Recommended Crop: **{prediction[0]}**")

elif option == "Monitorization":
    st.title("Monitorization")

    # Select crop from the dataset
    crop = st.selectbox("Select Crop", df['label'].unique())

    # Input parameters for the selected crop
    nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, step=0.1)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0, step=0.1)
    potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, step=0.1)
    temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, step=1)
    pH = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)

    # Prepare input for parameter checking
    params = {
        'Nitrogen': nitrogen,
        'Phosphorus': phosphorus,
        'Potassium': potassium,
        'Temperature': temp,
        'Humidity': humidity,
        'pH': pH
    }

    # Check and display parameter adjustments
    if st.button("Check Parameters"):
        adjustments = check_parameters(params, df, crop)
        if "error" in adjustments:
            st.write(adjustments["error"])
        else:
            st.write("Parameter Adjustments Needed:")
            for param, message in adjustments.items():
                st.write(message)
