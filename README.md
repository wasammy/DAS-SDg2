# Underground Nuclear Explosion (UNE) Detection via Distributed Acoustic Sensing (DAS)

---

## Overview
Applying DAS to UNE monitoring introduces a challenge, as all historical nuclear test recordings exist as seismometer velocity data, not as the strain-rate measurements compatible with DAS systems. This project's research foundation has been established through reviewing relevant literature spanning seismic wave physics, nuclear monitoring systems, DAS signal processing, and machine learning-based event classification. A basic data processing system was built and validated on two real DAS earthquake datasets: the 2016 M5.8 Pawnee, Oklahoma earthquake (SGY format), and the 2023 M5.0 Cook Inlet, Alaska earthquake (HDF5 format), demonstrating that the system correctly handles different file formats, removes faulty sensors, isolates seismic frequencies through filtering and automatically detects seismic events using the STA/LTA algorithm.

This repository houses the data processing and transformation scripts used to convert raw seafloor cable velocity data into strain-rate data.

---

## Quick Breakdown

* **SEGY Data Processing:** Processed large-scale geophysical data files mapping seismic wave propagation across seafloor cables.
* **Signal Visualization:** Mapped P-wave and S-wave arrival patterns using apparent velocity profiles.
* **Frequency Filtering:** Applied bandpass filters to isolate target frequencies and remove environmental background noise.
* **Automated Event Detection:** Implemented STA/LTA triggering algorithms to flag seismic events.

---

## Repository Structure
```text
DAS-SDg2/
│
├── notebooks/
│   └── 01_data_loading.ipynb    # Data loading, filtering, and initial processing
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

