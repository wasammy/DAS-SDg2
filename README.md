# Underground Nuclear Explosion (UNE) Detection via Distributed Acoustic Sensing (DAS)
ME 473 & ME 474 (Senior Design): Group 2
---

## Overview
DAS turns ordinary fiber-optic cable into thousands of seismic sensors, which makes it attractive for monitoring underground nuclear explosions (UNEs). The challenge is that every historical nuclear test was recorded on conventional seismometers, not DAS. There is no real DAS recording of a nuclear explosion to train on.
 
This project builds a detection and classification pipeline that takes raw DAS data, finds seismic events, and labels them as an earthquake or a possible underground nuclear explosion.

---

## Status

**Spring 2026 (complete!):** Built and validated a DAS processing pipeline on two real earthquake datasets: the 2016 M5.8 Pawnee, Oklahoma earthquake (SEG-Y, Stanford DAS array) and the 2023 M5.0 Cook Inlet, Alaska earthquake (HDF5, ~81 km seafloor cable). The pipeline handles both file formats, removes faulty channels, bandpass filters, and detects events with STA/LTA.

**Fall 2026 (in progress):** Following the approach in our advisor's EQ/UNE machine learning evaluation:
1. Train Random Forest (RF-v2 feature set) and CNN classifiers on seismometer data: UNEs from GTUNE, earthquakes from CREW.
2. Train separate models by epicentral distance (<5°, 5–10°, 10–20°), since classification is most accurate within a distance range.
3. Make DAS data look like the training data (P-aligned 60 s windows, 0.5–10 Hz, 20 Hz sampling, normalized) and test whether the classifiers transfer to DAS.
4. Test explosion-like signals on DAS, which no prior work has done!


---

## Repo Structure

```text
DAS-SDg2/
├── src/          # reusable Python code (loading, preprocessing, features, models)
├── notebooks/    # Jupyter notebooks
│   └── 01_das_signal_processing.ipynb   # Spring 2026 work: loading, filtering, STA/LTA
├── scripts/      # helper scripts
├── data/         # local data only
├── docs/         # reports and test plans... maybe more
├── assets/       # figures used
├── DATA.md       # origin of datasets we use
└── requirements.txt
```
 ## Setup

```bash
git clone https://github.com/wasammy/DAS-SDg2.git
cd DAS-SDg2
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```
Then copy any remaining data files from the shared Drive into `data/` (see `DATA.md`), and run `notebooks/01_das_signal_processing.ipynb` to check everything works.

## Important References
- Lindsey et al. (2017), Fiber-optic network observations of earthquake wavefields, GRL.
- Barama et al. (2022), GTUNE: An assembled global seismic dataset of underground nuclear test blasts, SRL.
- Barama et al. (2023), Global nuclear explosion discrimination using a convolutional neural network, GRL.
- Aguilar Suarez & Beroza (2024), Curated Regional Earthquake Waveforms (CREW) dataset, Seismica.
- Zhu et al. (2023), Seismic arrival-time picking on DAS data using semi-supervised learning, Nature Communications.