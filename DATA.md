# Data Sources
data files too large to store on here; keep copy in local /data folder instead.

## Current datasets

- **Monterey Bay DAS** (`data/monterey_bay/`): 33 submarine DAS earthquake
  waterfalls, ~137 MB each, ~4.5 GB total. The main DAS earthquake corpus
  for this semester.
    - Source: https://huggingface.co/datasets/AI4EPS/quakeflow_das (`monterey_bay/data`)
    - Download with: `python scripts/download_monterey.py`

- **TERRA.h5**: 2023 M5.0 Cook Inlet, Alaska earthquake; recorded on the TERRA fiber cable (about 81 km, seafloor). HDF5 format.
    - Source: https://dasway.ess.washington.edu/gci/events/2023-07-28/11726446/TERRA.h5
- **cbt_processed_20160903_120154.932+0000.sgy**: 2016 M5.8 Pawnee, Oklahoma earthquake, recorded on the Stanford University campus DAS array (referenced in Lindsey et al. 2017). SEG-Y format.
    - Source: https://github.com/eileenrmartin/FiberOpticEarthquakes/tree/master/Data/Pawnee
    
## Planned datasets

- **GTUNE**: seismometer recordings of underground nuclear tests (no DAS).
  Prepared download is 4 HDF5 pandas tables, ~543 MB: FDSN, LASA, LLNL, FSU.
  Known issues: 16,645 records but ~3,655 are empty; sampling rates range
  8-100 Hz (anything under 20 Hz is unusable for the 0.5-10 Hz band);
  traces are NOT P-aligned despite the README saying they are, so P picks
  have to be made with STA/LTA. Only 734 FDSN records fall within 20 degrees.
    - Source: https://github.com/lbarama/GTUNE (waveforms via the Google Drive
      link in Prepared_datasets/HDF5_prepared_data)
      
- **CREW**: Curated Regional Earthquake Waveforms dataset.
    - Source:accessed through the SeisBench Python package (seisbench.data.CREW). The paper is Aguilar Suarez & Beroza (2024), in Seismica.

## Shared team Drive

[WILL ADD LATER]