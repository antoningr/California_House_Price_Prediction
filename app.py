# app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from api import predict_house_price, get_prediction_history

st.set_page_config(page_title="California House Price Predictor", layout="wide")

st.title("🏠 California House Price Prediction App")
st.markdown("Predict median house values using a trained model based on California Housing data.")

# Mode selection sidebar
st.sidebar.markdown("**Input Tips (Data Summary)**")
st.sidebar.markdown("""
Dataset from 1990 US census, 20,640 block groups.
- `Median Income`: in 10k USD, typical range 1 to 15.
- `House Age`: usually 1 to 52 years.
- `Average Rooms per House`: rooms per household.
- `Average Bedrooms per House`: less than rooms.
- `Block Population`: total block group population, usually 100–10,000
- `Average Household Size`: usually <5, warning above 10
- `Latitude` / `Longitude`: geographic coordinates in California.
""")

# Initialize session state for results
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "inputs" not in st.session_state:
    st.session_state.inputs = None

with st.form("prediction_form"):
    st.subheader("🏡 Input House Characteristics")

    col1, col2 = st.columns(2)
    with col1:
        MedInc = st.number_input("Median Income (in 10k)", min_value=0.0, value=5.0)
        HouseAge = st.number_input("House Age (years)", min_value=1, max_value=100, value=20)
        AveRooms = st.number_input("Average Rooms per House", min_value=1.0, value=5.0)
        AveBedrms = st.number_input("Average Bedrooms per House", min_value=0.5, value=1.0)

    with col2:
        Population = st.number_input("Block Population", min_value=1, value=1000)
        AveOccup = st.number_input("Average Household Size", min_value=0.5, value=3.0)
        Latitude = st.slider("Latitude", 32.5, 42.0, 34.0)
        Longitude = st.slider("Longitude", -124.0, -114.0, -118.0)

    if AveOccup > 10:
        st.warning("⚠️ Average occupancy is unusually high. Please double-check.")

    submitted = st.form_submit_button("Predict Price 💰")

if submitted:
    # Save inputs and prediction to session state
    user_input = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
    }
    st.session_state.inputs = user_input
    st.session_state.prediction = predict_house_price(user_input)

# Show prediction and map only if prediction exists
if st.session_state.prediction is not None and st.session_state.inputs is not None:
    price = st.session_state.prediction * 100000
    formatted_price = f"${price:,.0f}"  # Adds commas and rounds to integer
    st.success(f"🏡 **Predicted Median House Price:** {formatted_price}")

    st.subheader("📍 House Location Map")
    lat = st.session_state.inputs["Latitude"]
    lon = st.session_state.inputs["Longitude"]

    location_map = folium.Map(location=[lat, lon], zoom_start=9)
    folium.Marker(
        [lat, lon],
        popup=f"Price: ${price:.2f}",
        tooltip="Property Location",
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(location_map)
    st_folium(location_map, width=700, height=400)

st.markdown("---")
st.subheader("🧾 Prediction History")
history_df = get_prediction_history()
if not history_df.empty:
    st.dataframe(history_df.tail(10), use_container_width=True)
else:
    st.info("No predictions have been made yet.")
