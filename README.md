# 🧲 Microwave Resonator Peak Analyzer

An interactive web application built with Streamlit and Python for automated analysis of microwave resonator frequency sweeps across varying temperatures.


---

## 🌟 Features

* **Multi-File Processing:** Upload multiple `.txt` or `.csv` frequency sweep data files simultaneously.
* **Automatic Temperature Extraction:** Automatically parses temperature values (`T_X.XXX`) directly from file names.
* **Frequency Band Detection:** Splits full-spectrum frequency sweeps into distinct resonant bands automatically using adjustable gap factors.
* **Savitzky-Golay Signal Smoothing:** Filters out high-frequency noise without broadening or shifting peak locations.
* **Automated Peak Finding:** Detects resonance peaks based on configurable prominence thresholds.
* **Interactive Visualizations:**
  * **Sweep Colormaps:** Plot full frequency spectra across temperatures using custom color gradients.
  * **$f$ vs $T$ Trends:** Track extracted resonance frequencies across temperatures per band.
* **Data Export:** Export extracted resonance data ($f$ vs $T$) per band directly to CSV format.
* **Demo Data Included:** Built-in demo file support for quick testing without uploading personal datasets.

---

## 📁 Data Format Requirements

Your input text files should be structured as comma-separated values (CSV) containing at least the following header columns:

| Column Name | Description |
| :--- | :--- |
| **`Frequency`** | Frequency values in Hz |
| **`LinMag`** | Linear magnitude of the resonator signal |

### File Naming Convention
To enable automatic temperature parsing, include `T_` followed by the temperature value in your filename:
```text
test_1500_P_-16.0_T_3.664.txt --> Parsed Temperature: 3.664 K
