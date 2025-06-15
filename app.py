import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import datetime

# --- Konfigurasi Halaman (Harus menjadi perintah pertama Streamlit) ---
st.set_page_config(page_title="Prediksi Keterlambatan Penerbangan", layout="wide")

# --- Pemuatan Artefak ---
@st.cache_data
def load_artifacts():
    """Memuat semua artefak yang dibutuhkan dari file."""
    model_arrival = joblib.load('model_arrival.pkl')
    model_departure = joblib.load('model_departure.pkl')
    scaler = joblib.load('scaler.pkl')
    
    with open('freq_maps.json', 'r') as f:
        freq_maps = json.load(f)
    
    with open('final_columns.json', 'r') as f:
        final_columns = json.load(f)

    with open('median_values.json', 'r') as f:
        median_values = json.load(f)
        
    with open('unique_values.json', 'r') as f:
        unique_values = json.load(f)

    return model_arrival, model_departure, scaler, freq_maps, final_columns, median_values, unique_values

# Memuat semua file di awal
model_arr, model_dep, scaler, freq_maps, final_cols, median_vals, unique_vals = load_artifacts()

# --- Antarmuka Pengguna (UI) ---
st.title("✈️ Prediksi Keterlambatan Penerbangan")
st.write("Aplikasi ini memprediksi keterlambatan keberangkatan dan kedatangan pesawat.")

st.sidebar.header("Masukkan Detail Penerbangan:")

# Input Kategori
airline = st.sidebar.selectbox("Maskapai (Airline)", sorted(unique_vals['Airline']))
origin = st.sidebar.selectbox("Bandara Asal (Origin)", sorted(unique_vals['Origin']))
dest = st.sidebar.selectbox("Bandara Tujuan (Dest)", sorted(unique_vals['Dest']))

st.sidebar.markdown("---")

# Input Waktu dengan st.time_input
crs_dep_time_input = st.sidebar.time_input("Jadwal Keberangkatan", datetime.time(13, 25))
dep_time_input = st.sidebar.time_input("Waktu Keberangkatan Aktual", datetime.time(13, 35))
wheels_off_input = st.sidebar.time_input("Waktu Roda Lepas Landas", datetime.time(13, 45))
crs_arr_time_input = st.sidebar.time_input("Jadwal Kedatangan", datetime.time(15, 10))
wheels_on_input = st.sidebar.time_input("Waktu Roda Mendarat", datetime.time(14, 55))
arr_time_input = st.sidebar.time_input("Waktu Kedatangan Aktual", datetime.time(15, 0))

st.sidebar.markdown("---")

# Input Durasi dengan st.number_input
taxi_out = st.sidebar.number_input("Waktu Taxi Out (menit)", min_value=1, max_value=200, value=16)
taxi_in = st.sidebar.number_input("Waktu Taxi In (menit)", min_value=1, max_value=200, value=7)
air_time = st.sidebar.number_input("Waktu di Udara (menit)", min_value=10, max_value=800, value=107)


# Tombol untuk melakukan prediksi
if st.button("🚀 Prediksi Keterlambatan"):

    # --- Konversi Input Ramah Pengguna ke Format Numerik untuk Model ---
    crs_dep_time = crs_dep_time_input.hour * 100 + crs_dep_time_input.minute
    dep_time = dep_time_input.hour * 100 + dep_time_input.minute
    wheels_off = wheels_off_input.hour * 100 + wheels_off_input.minute
    crs_arr_time = crs_arr_time_input.hour * 100 + crs_arr_time_input.minute
    wheels_on = wheels_on_input.hour * 100 + wheels_on_input.minute
    arr_time = arr_time_input.hour * 100 + arr_time_input.minute
    
    # --- Preprocessing Data Input ---
    # 1. Buat DataFrame dari input dan nilai median
    input_data = median_vals.copy()
    
    input_data.update({
        'Airline': airline, 'Origin': origin, 'Dest': dest,
        'CRSDepTime': crs_dep_time, 'DepTime': dep_time, 'TaxiOut': taxi_out,
        'TaxiIn': taxi_in, 'WheelsOff': wheels_off, 'WheelsOn': wheels_on,
        'CRSArrTime': crs_arr_time, 'ArrTime': arr_time, 'AirTime': air_time
    })

    # 2. Feature Engineering
    input_data['FlightDuration'] = input_data['TaxiOut'] + input_data['TaxiIn']

    # Buat DataFrame
    input_df_raw = pd.DataFrame([input_data])
    
    # 3. Encoding Kolom Kategorikal
    input_encoded = input_df_raw.copy()
    for col in ['Airline', 'Origin', 'Dest']:
        input_encoded[col] = input_encoded[col].map(freq_maps[col]).fillna(0)
    
    # 4. Normalisasi Kolom Numerik
    numeric_cols_to_scale = [col for col in median_vals.keys() if col in scaler.feature_names_in_]
    input_numeric_scaled = scaler.transform(input_encoded[numeric_cols_to_scale])
    input_numeric_scaled_df = pd.DataFrame(input_numeric_scaled, columns=numeric_cols_to_scale, index=input_encoded.index)
    
    # Gabungkan kembali
    input_processed = pd.concat([
        input_encoded[['Airline', 'Origin', 'Dest']].reset_index(drop=True),
        input_numeric_scaled_df
    ], axis=1)

    # 5. Pastikan semua kolom final ada dan urutannya benar
    input_final = pd.DataFrame(columns=final_cols)
    input_final = pd.concat([input_final, input_processed], ignore_index=True).fillna(0)
    input_final = input_final[final_cols]
    
    # --- Melakukan Prediksi ---
    pred_departure = model_dep.predict(input_final)[0]
    pred_arrival = model_arr.predict(input_final)[0]

    # --- Menampilkan Hasil ---
    st.subheader("Hasil Prediksi:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Keterlambatan Keberangkatan", f"{pred_departure:.2f} menit")
        if pred_departure > 15:
            st.warning("Pesawat diperkirakan **terlambat** berangkat.")
        elif pred_departure < -5:
            st.success("Pesawat diperkirakan berangkat **lebih awal**.")
        else:
            st.info("Pesawat diperkirakan berangkat **tepat waktu**.")

    with col2:
        st.metric("Keterlambatan Kedatangan", f"{pred_arrival:.2f} menit")
        if pred_arrival > 15:
            st.error("Pesawat diperkirakan **terlambat** tiba.")
        elif pred_arrival < -5:
            st.success("Pesawat diperkirakan tiba **lebih awal**.")
        else:
            st.info("Pesawat diperkirakan tiba **tepat waktu**.")
            
    with st.expander("Lihat Data Input yang Telah Diproses"):
        st.write("Data ini adalah representasi final yang diberikan ke model setelah encoding dan scaling.")
        st.dataframe(input_final)
