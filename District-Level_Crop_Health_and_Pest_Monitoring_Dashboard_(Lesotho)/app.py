import streamlit as st  # type: ignore
import datetime
import geopandas as gpd  # type: ignore
from scripts.gee_functions import (
    get_ndvi, get_soil_moisture, get_precipitation,
    get_temperature, get_humidity, get_irradiance, get_simulated_hyperspectral
)
import geemap  # type: ignore
import ee  # type: ignore
from shapely.geometry import mapping  # type: ignore
import joblib  # type: ignore

st.set_page_config(layout="wide")
st.title("🌱 District-Level Crop Health & Pest Monitoring Dashboard (Lesotho)")

start_date = st.date_input("Start Date", datetime.date(2024, 11, 1))
end_date = st.date_input("End Date", datetime.date(2025, 5, 31))

shp_path = r'C:\\Users\\MY PC\\Documents\\GIS DATA\\BOUNDARIES\\LSO_adm\\LSO_adm1.shp'
district_gdf = gpd.read_file(shp_path)
district_names = district_gdf['NAME_1'].tolist()
selected_district = st.selectbox("Select District", district_names)

try:
    disease_model = joblib.load("models/disease_risk_model.pkl")
    pest_model = joblib.load("models/pest_risk_model.pkl")
except:
    disease_model = None
    pest_model = None

district_geom = district_gdf[district_gdf['NAME_1'] == selected_district].geometry.values[0]

if st.button("Generate District Map"):
    with st.spinner("Fetching satellite data for selected district..."):
        ee.Initialize(project='winged-tenure-464005-p9')
        geom = ee.Geometry(mapping(district_geom))

        ndvi = get_ndvi(str(start_date), str(end_date), geom)
        soil = get_soil_moisture(str(start_date), str(end_date), geom)
        precip = get_precipitation(str(start_date), str(end_date), geom)
        temp = get_temperature(str(start_date), str(end_date), geom)
        humidity = get_humidity(str(start_date), str(end_date), geom)
        irradiance = get_irradiance(str(start_date), str(end_date), geom)
        hyper = get_simulated_hyperspectral(str(start_date), str(end_date), geom)

        b5 = hyper.select("B5")
        b6 = hyper.select("B6")
        b7 = hyper.select("B7")
        b11 = hyper.select("B11")
        b12 = hyper.select("B12")

        stats = ee.Image.cat([
            ndvi.rename('NDVI'),
            soil.rename('Soil'),
            precip.rename('Precip'),
            temp.rename('Temp'),
            humidity.rename('Humidity'),
            irradiance.rename('Irradiance'),
            b5.rename('b5'), b6.rename('b6'), b7.rename('b7'), b11.rename('b11'), b12.rename('b12')
        ]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=30,
            maxPixels=1e9
        ).getInfo()

        mean_ndvi = stats.get('NDVI', 0)
        b5_val = stats.get('b5', 0)
        b6_val = stats.get('b6', 0)
        b7_val = stats.get('b7', 0)
        b11_val = stats.get('b11', 0)
        b12_val = stats.get('b12', 0)

        disease_risk = 'N/A'
        pest_risk = 'N/A'

        if disease_model:
            disease_risk = disease_model.predict([[b5_val, b6_val, b7_val, b11_val, b12_val, mean_ndvi]])[0]

        if pest_model:
            pest_risk = pest_model.predict([[b5_val, b6_val, b7_val, b11_val, b12_val, mean_ndvi]])[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Numerical Summary")
            st.metric("Mean NDVI", round(mean_ndvi, 3))
            st.metric("Soil Moisture", round(stats.get('Soil', 0), 3))
            st.metric("Precipitation (mm)", round(stats.get('Precip', 0), 2))
            st.metric("Temperature (K)", round(stats.get('Temp', 0), 2))
            st.metric("Humidity (K)", round(stats.get('Humidity', 0), 2))
            st.metric("Irradiance (W/m²)", round(stats.get('Irradiance', 0), 2))
            st.metric("Disease Risk", disease_risk)
            st.metric("Pest Risk", pest_risk)

        with col2:
            Map = geemap.Map()
            Map.centerObject(geom, 9)
            Map.addLayer(ndvi, {'min': 0, 'max': 1, 'palette': ['brown', 'yellow', 'green']}, 'NDVI')
            Map.addLayer(soil, {'min': 0, 'max': 0.5}, 'Soil Moisture')
            Map.addLayer(precip, {'min': 0, 'max': 200}, 'Precipitation')
            Map.addLayer(temp, {'min': 270, 'max': 310}, 'Temperature (K)')
            Map.addLayer(humidity, {}, 'Humidity')
            Map.addLayer(irradiance, {}, 'Irradiance')

            Map.to_html('outputs/map_exports/district_map.html')
            st.success("Map created!")
            st.components.v1.iframe("outputs/map_exports/district_map.html", height=600, scrolling=True)
