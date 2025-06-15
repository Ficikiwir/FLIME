import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- Konfigurasi Halaman (Harus menjadi perintah pertama Streamlit) ---
st.set_page_config(page_title="Prediksi Keterlambatan Penerbangan", layout="wide")

# --- Pemuatan Artefak ---
# Menggunakan cache agar model dan preprocessor tidak perlu dimuat ulang setiap kali ada interaksi
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
st.write("Aplikasi ini memprediksi keterlambatan keberangkatan dan kedatangan pesawat berdasarkan beberapa fitur kunci.")

# Input dari pengguna di sidebar
st.sidebar.header("Masukkan Detail Penerbangan:")

# Berdasarkan feature importance, kita pilih beberapa fitur utama
# Untuk fitur kategorikal, kita gunakan data unik yang sudah disimpan
airline = st.sidebar.selectbox("Maskapai (Airline)", sorted(unique_vals['Airline']))
origin = st.sidebar.selectbox("Bandara Asal (Origin)", sorted(unique_vals['Origin']))
dest = st.sidebar.selectbox("Bandara Tujuan (Dest)", sorted(unique_vals['Dest']))

# Fitur numerik paling penting
# Menggunakan nilai median sebagai default
crs_dep_time = st.sidebar.slider("Jadwal Keberangkatan (CRS DepTime)", 0, 2359, int(median_vals.get('CRSDepTime', 1325)))
dep_time = st.sidebar.slider("Waktu Keberangkatan Aktual (DepTime)", 0, 2359, int(median_vals.get('DepTime', 1335)))
taxi_out = st.sidebar.slider("Waktu Taxi Out (menit)", 1, 150, int(median_vals.get('TaxiOut', 16)))
taxi_in = st.sidebar.slider("Waktu Taxi In (menit)", 1, 150, int(median_vals.get('TaxiIn', 7)))
wheels_off = st.sidebar.slider("Waktu Roda Lepas Landas (WheelsOff)", 0, 2359, int(median_vals.get('WheelsOff', 1345)))
wheels_on = st.sidebar.slider("Waktu Roda Mendarat (WheelsOn)", 0, 2359, int(median_vals.get('WheelsOn', 1455)))
crs_arr_time = st.sidebar.slider("Jadwal Kedatangan (CRS ArrTime)", 0, 2359, int(median_vals.get('CRSArrTime', 1510)))
arr_time = st.sidebar.slider("Waktu Kedatangan Aktual (ArrTime)", 0, 2359, int(median_vals.get('ArrTime', 1500)))
air_time = st.sidebar.slider("Waktu di Udara (AirTime)", 10, 700, int(median_vals.get('AirTime', 107)))


# Tombol untuk melakukan prediksi
if st.button("🚀 Prediksi Keterlambatan"):
    
    # --- Preprocessing Data Input ---
    # 1. Buat DataFrame dari input dan nilai median
    # Kita mulai dengan semua nilai median, lalu timpa dengan input pengguna
    input_data = median_vals.copy()
    
    input_data['Airline'] = airline
    input_data['Origin'] = origin
    input_data['Dest'] = dest
    input_data['CRSDepTime'] = crs_dep_time
    input_data['DepTime'] = dep_time
    input_data['TaxiOut'] = taxi_out
    input_data['TaxiIn'] = taxi_in
    input_data['WheelsOff'] = wheels_off
    input_data['WheelsOn'] = wheels_on
    input_data['CRSArrTime'] = crs_arr_time
    input_data['ArrTime'] = arr_time
    input_data['AirTime'] = air_time

    # 2. Feature Engineering
    input_data['FlightDuration'] = input_data['TaxiOut'] + input_data['TaxiIn']

    # Buat DataFrame
    input_df_raw = pd.DataFrame([input_data])
    
    # 3. Encoding Kolom Kategorikal
    input_encoded = input_df_raw.copy()
    for col in ['Airline', 'Origin', 'Dest']:
        # Menggunakan map yang sudah disimpan. Jika ada nilai baru, isi dengan 0
        input_encoded[col] = input_encoded[col].map(freq_maps[col]).fillna(0)
    
    # 4. Normalisasi Kolom Numerik
    numeric_cols_to_scale = [col for col in median_vals.keys() if col in scaler.feature_names_in_]
    
    # Pastikan hanya kolom yang ada di scaler yang diubah
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
    input_final = input_final[final_cols] # Urutkan kolom sesuai training
    
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
