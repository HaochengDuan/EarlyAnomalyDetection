# Early Traffic Anomaly Detection

Code for "Know Unreported Roadway Incidents in Real-time: Early Traffic Anomaly Detection".

The speed data and incident reports used in the paper are commercial data and cannot be released under their license agreements. `data/synthetic_link` is a reasonably constructed synthetic dataset for one target link and its sub-graph; it gives results similar to those in the paper.

## Run

```bash
pip install -r requirements.txt
bash scripts/run_all.sh
```

Results are written to `results/table3.md`, `results/table5.md`, `results/table6.md` and `results/training_loss.png`.

`CKPT=checkpoints bash scripts/run_all.sh` uses the trained Transformers in `checkpoints/` instead of training them.

## Data

| File | Columns |
|---|---|
| `links.csv` | `link_id, order, length_mi, upstream_link, downstream_link` |
| `speed_5min.csv` | `timestamp`, one speed column (mph) per link |
| `incident_reports.csv` | `report_id, link_id, start_time, end_time` |

## Raw data

`raw_data_processor/process.py` builds the three files from a TMC GeoJSON (`tmc, miles, start_latitude, start_longitude, end_latitude, end_longitude`), INRIX speed exports (`tmc_code, measurement_tstamp, speed`) and RITIS incident exports (`Start time, Closed time, Latitude, Longitude`).

```bash
pip install -r raw_data_processor/requirements.txt
python raw_data_processor/process.py --tmc-shape tmc.geojson --target <tmc> --speed speed.csv --incidents incidents.csv --start 2022-02-14 --end 2023-02-12 --out data/my_link
DATA=data/my_link LINK=<tmc> bash scripts/run_all.sh
```
