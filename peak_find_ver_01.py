import streamlit as st
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Microwave Resonator Analysis", layout="wide")
st.title("🧲 Microwave Resonator Peak Analysis")

# ===== SIDEBAR: USER INPUTS =====
st.sidebar.header("1. Upload Data")
# Accepts multiple text files directly from your PC





#uploaded_files = st.sidebar.file_uploader("Upload all sweep .txt files", accept_multiple_files=True, type=["txt", "csv"])
# Place this right after st.sidebar.
#if st.sidebar.button("🗑️ Reset / Clear All Data"):
 #   st.rerun()

# ===== SIDEBAR: USER INPUTS =====
st.sidebar.header("1. Upload Data")

# Initialize a state key for the uploader if it doesn't exist
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# Pass the dynamic key to the uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload all sweep .txt files", 
    accept_multiple_files=True, 
    type=["txt", "csv"],
    key=f"uploader_{st.session_state['uploader_key']}"
)

# Reset Button: Incrementing the key forces Streamlit to rebuild a clean uploader widget
if st.sidebar.button("🗑️ Clear All Files", use_container_width=True):
    st.session_state["uploader_key"] += 1
    st.rerun()











st.sidebar.header("2. Analysis Parameters")
GAP_FACTOR = st.sidebar.number_input("Gap Factor (Band splitting)", value=100, step=10)
SMOOTH_WINDOW = st.sidebar.number_input("Smoothing Window (Must be odd)", value=5, step=2, min_value=3)
SMOOTH_POLY = st.sidebar.number_input("Smoothing Polynomial", value=3, step=1, min_value=1, max_value=SMOOTH_WINDOW-1)
PROMINENCE = st.sidebar.number_input("Minimum Peak Prominence", value=0.001, step=0.0005, format="%.4f")

PLOT_INDIVIDUAL = st.sidebar.checkbox("Show individual peak plots", value=False)

# ===== FUNCTIONS (Adapted from your code) =====
def extract_temperature(filename):
    match = re.search(r"T_([0-9.]+)", filename)
    if match:
        return float(match.group(1).rstrip("."))
    return None

def detect_frequency_bands(frequency):
    df = np.diff(frequency)
    threshold = GAP_FACTOR * np.median(df)
    splits = [0]
    for i in range(1, len(frequency)):
        if (frequency[i] - frequency[i-1]) > threshold:
            splits.append(i)
    splits.append(len(frequency))
    return splits

def smooth(y):
    if len(y) < SMOOTH_WINDOW:
        return y
    return savgol_filter(y, SMOOTH_WINDOW, SMOOTH_POLY)

# ===== MAIN APP LOGIC =====
if uploaded_files:
    band_results = {}
    band_full_data = {}
    
    # Extract temperatures and sort files
    files_with_T = []
    for file in uploaded_files:
        T = extract_temperature(file.name)
        if T is not None:
            files_with_T.append((T, file))
            
    if not files_with_T:
        st.error("Could not extract temperatures from filenames. Ensure files have 'T_X.XXX' in the name.")
        st.stop()
        
    files_with_T = sorted(files_with_T, key=lambda x: x[0])
    
    with st.spinner("Analyzing files..."):
        for T, file in files_with_T:
            # Reset file pointer for reading
            file.seek(0)
            data = pd.read_csv(file, sep=",")
            data.columns = [col.strip() for col in data.columns]
            
            f = data["Frequency"].values
            linmag = data["LinMag"].values
            
            splits = detect_frequency_bands(f)
            
            for band_idx in range(len(splits)-1):
                start = splits[band_idx]
                end = splits[band_idx+1]
                
                f_band = f[start:end]
                lm_band = linmag[start:end]
                lm_smooth = smooth(lm_band)
                
                # Store full data
                if band_idx not in band_full_data:
                    band_full_data[band_idx] = []
                    
                band_full_data[band_idx].append({
                    "T": T,
                    "f": f_band / 1e9,      # GHz
                    "linmag": lm_band,
                    "smooth": lm_smooth,
                    "filename": file.name
                })
                
                peaks, _ = find_peaks(lm_smooth, prominence=PROMINENCE)
                
                if len(peaks) == 0:
                    continue
                    
                # Choose strongest peak
                peak_idx = peaks[np.argmax(lm_smooth[peaks])]
                f0 = f_band[peak_idx] / 1e9  # GHz
                
                if band_idx not in band_results:
                    band_results[band_idx] = {"T": [], "f": []}
                    
                band_results[band_idx]["T"].append(T)
                band_results[band_idx]["f"].append(f0)
                
                # Optional: Plot Individual scans
                if PLOT_INDIVIDUAL:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(f_band/1e9, lm_band, color='blue', alpha=0.5, label="Raw")
                    ax.plot(f_band/1e9, lm_smooth, color='orange', label="Smooth")
                    ax.axvline(f0, color='red', linestyle="--")
                    ax.text(f0+0.001, lm_smooth[peak_idx], f"f = {f0:.6f} GHz\nT = {T:.3f} K", color="red")
                    ax.set_xlabel("Frequency (GHz)")
                    ax.set_ylabel("LinMag")
                    ax.set_title(f"{file.name} | Band {band_idx+1}")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig) # Keep memory clean

        st.success(f"Successfully processed {len(files_with_T)} sweeps across {len(band_full_data)} frequency bands.")

        # ============================================================
        # VISUALIZATION: TABS FOR CLEAN ORGANIZATION
        # ============================================================
        tab1, tab2 = st.tabs(["🌈 Sweep Colormaps", "📈 f vs T Trends"])
        
        with tab1:
            st.subheader("Frequency Sweeps Over Temperature")
            cols = st.columns(min(len(band_full_data), 3)) # Max 3 plots per row
            
            for idx, (band_idx, dataset) in enumerate(band_full_data.items()):
                fig, ax = plt.subplots(figsize=(6, 5))
                dataset = sorted(dataset, key=lambda x: x["T"])
                temps = np.array([entry["T"] for entry in dataset])
                
                norm = mcolors.Normalize(vmin=temps.min(), vmax=temps.max())
                cmap = cm.viridis
                
                for entry in dataset:
                    color = cmap(norm(entry["T"]))
                    ax.plot(entry["f"], entry["linmag"], color=color, linewidth=0.8)
                    
                ax.set_xlabel("Frequency (GHz)")
                ax.set_ylabel("LinMag")
                ax.set_title(f"Band {band_idx + 1}")
                ax.grid(True, alpha=0.3)
                
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                fig.colorbar(sm, ax=ax, label="Temperature (K)")
                
                cols[idx % 3].pyplot(fig)
                plt.close(fig)

        with tab2:
            st.subheader("Extracted Resonance vs Temperature")
            cols = st.columns(min(len(band_results), 3))
            
            for idx, band_idx in enumerate(band_results):
                if len(band_results[band_idx]["T"]) == 0:
                    continue
                    
                T_vals = band_results[band_idx]["T"]
                f_vals = band_results[band_idx]["f"]
                
                sorted_pairs = sorted(zip(T_vals, f_vals))
                T_sorted, f_sorted = zip(*sorted_pairs)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(T_sorted, f_sorted, marker='o', linestyle='-', color='#d62728')
                ax.set_xlabel("Temperature (K)")
                ax.set_ylabel("Resonance Frequency (GHz)")
                ax.set_title(f"Band {band_idx+1}")
                ax.grid(True, linestyle="--", alpha=0.7)
                
                cols[idx % 3].pyplot(fig)
                plt.close(fig)
                
                # Add a download button for this specific band's results
                csv_data = pd.DataFrame({"Temperature (K)": T_sorted, "Frequency (GHz)": f_sorted}).to_csv(index=False)
                cols[idx % 3].download_button(
                    label=f"Download Band {band_idx+1} Results",
                    data=csv_data,
                    file_name=f"Band_{band_idx+1}_f_vs_T.csv",
                    mime="text/csv"
                )
else:
    st.info("👈 Upload your data folder files in the sidebar to begin processing.")