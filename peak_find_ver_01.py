# ─────────────────────────────────────────────────────────────────────────────
#  📈  Microwave Resonator Peak Analysis – fully parallelised version
# ─────────────────────────────────────────────────────────────────────────────

# --------------------------------------------------------------
# 1️⃣  IMPORTS
# --------------------------------------------------------------
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import savgol_filter, find_peaks

# --------------------------------------------------------------
# 2️⃣  PAGE CONFIG & TITLE
# --------------------------------------------------------------
st.set_page_config(page_title="Microwave Resonator Analysis", layout="wide")
st.title("🧲 Microwave Resonator Peak Analysis")

# --------------------------------------------------------------
# 3️⃣  SIDEBAR – FILE UPLOADER + PARAMETERS
# --------------------------------------------------------------
st.sidebar.header("1. Upload Data")

# ---- a)  uploader with a “reset” key ---------------------------------
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

uploaded_files = st.sidebar.file_uploader(
    "Upload all sweep .txt files",
    accept_multiple_files=True,
    type=["txt", "csv"],
    key=f"uploader_{st.session_state['uploader_key']}",
)

if st.sidebar.button("🗑️ Clear All Files", use_container_width=True):
    st.session_state["uploader_key"] += 1
    st.rerun()

# ---- b)  analysis parameters -----------------------------------------
st.sidebar.header("2. Analysis Parameters")
GAP_FACTOR = st.sidebar.number_input(
    "Gap Factor (Band splitting)", value=100, step=10
)
SMOOTH_WINDOW = st.sidebar.number_input(
    "Smoothing Window (must be odd)", value=5, step=2, min_value=3
)
SMOOTH_POLY = st.sidebar.number_input(
    "Smoothing Polynomial", value=3, step=1, min_value=1, max_value=SMOOTH_WINDOW - 1
)
PROMINENCE = st.sidebar.number_input(
    "Minimum Peak Prominence", value=0.001, step=0.0005, format="%.4f"
)

PLOT_INDIVIDUAL = st.sidebar.checkbox("Show individual peak plots", value=False)

# --------------------------------------------------------------
# 4️⃣  PURE‑PYTHON / NUMPY HELPERS (must be top‑level for pickling)
# --------------------------------------------------------------
def extract_temperature(fname: str) -> float | None:
    """Extract temperature from a filename like  …_T_1.23_….txt."""
    m = re.search(r"T_([0-9.]+)", fname)
    return float(m.group(1).rstrip(".")) if m else None


def detect_frequency_bands_np(freq: np.ndarray, gap_factor: float) -> np.ndarray:
    """
    Vectorised version of the original band‑split routine.
    Returns the split indices: [0, idx1, idx2, …, len(freq)].
    """
    df = np.diff(freq)
    thr = gap_factor * np.median(df)
    splits = np.where(df > thr)[0] + 1
    return np.concatenate(([0], splits, [len(freq)]))


def smooth_signal(y: np.ndarray, win: int, poly: int) -> np.ndarray:
    """Savitzky‑Golay smoothing (fallback to raw data if too short)."""
    return y if len(y) < win else savgol_filter(y, win, poly)


def process_one_file(
    file_bytes: bytes,
    filename: str,
    params: Dict[str, Any],
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    **Runs in a separate process**.
    Returns:
        temperature (float)
        list of per‑band dictionaries:
            {
                "band_idx": int,
                "f_GHz": np.ndarray,
                "linmag": np.ndarray,
                "smooth": np.ndarray,
                "peak_fGHz": float,
                "peak_idx": int,
            }
    """
    # ----- read CSV from the raw bytes ---------------------------------
    buf = BytesIO(file_bytes)
    df = pd.read_csv(buf, sep=",")
    df.columns = [c.strip() for c in df.columns]

    f = df["Frequency"].values.astype(np.float64)
    linmag = df["LinMag"].values.astype(np.float64)

    # ----- band detection ------------------------------------------------
    splits = detect_frequency_bands_np(f, params["gap_factor"])

    bands = []
    for band_idx in range(len(splits) - 1):
        start, end = splits[band_idx], splits[band_idx + 1]
        f_band = f[start:end]
        lm_band = linmag[start:end]

        # smoothing
        lm_smooth = smooth_signal(lm_band, params["smooth_window"], params["smooth_poly"])

        # peak detection
        peaks, _ = find_peaks(lm_smooth, prominence=params["prominence"])
        if len(peaks) == 0:
            # No peak in this band → skip it
            continue
        # strongest peak
        strongest = peaks[np.argmax(lm_smooth[peaks])]
        peak_fGHz = f_band[strongest] / 1e9

        bands.append(
            {
                "band_idx": band_idx,
                "f_GHz": f_band / 1e9,
                "linmag": lm_band,
                "smooth": lm_smooth,
                "peak_fGHz": peak_fGHz,
                "peak_idx": strongest,
            }
        )

    temperature = extract_temperature(filename)
    return temperature, bands


# --------------------------------------------------------------
# 5️⃣  PARALLEL DRIVER (cached)
# --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_parallel_analysis(
    file_info: List[Tuple[str, bytes]],
    params: Dict[str, Any],
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, List[float]]]]:
    """
    Executes `process_one_file` for every uploaded file on a pool of workers.
    Returns two structures that are **exactly what the original script expected**:

    * `band_full_data[band_idx]` → list of dicts with raw + smooth data.
    * `band_results[band_idx]` → {"T": [...], "f": [...]}  (peak frequencies vs T)
    """
    band_full_data: Dict[int, List[Dict[str, Any]]] = {}
    band_results: Dict[int, Dict[str, List[float]]] = {}

    # ----------------------------------------------------------
    #   Use as many processes as there are physical cores
    # ----------------------------------------------------------
    max_workers = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_one_file, bts, fname, params): (fname, bts)
            for fname, bts in file_info
        }

        for fut in as_completed(future_to_file):
            fname, _ = future_to_file[fut]
            try:
                temperature, band_list = fut.result()
            except Exception as exc:
                st.error(f"⚠️  Problem processing *{fname}*: {exc}")
                continue

            if temperature is None:
                st.warning(f"⚠️  Could not read temperature from *{fname}* – skipping.")
                continue

            for band in band_list:
                idx = band["band_idx"]

                # ----- store the full trace (for the colormap view) ----------
                if idx not in band_full_data:
                    band_full_data[idx] = []
                band_full_data[idx].append(
                    {
                        "T": temperature,
                        "f": band["f_GHz"],
                        "linmag": band["linmag"],
                        "smooth": band["smooth"],
