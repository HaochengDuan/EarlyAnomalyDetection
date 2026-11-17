import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

START_MIN, END_MIN, STEP_MIN = 330, 1260, 5

def bearing(line):
    xy = shapely.get_coordinates(line)
    return math.atan2(xy[-1, 1] - xy[0, 1], xy[-1, 0] - xy[0, 0])


def point_key(lat, lon):
    return lat.round(5).map("{:.5f}".format) + "," + lon.round(5).map("{:.5f}".format)


def turn(a, b):
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def corridor(tmc, target, upstream_mi, downstream_mi):
    tmc = tmc.set_index("tmc")
    start = point_key(tmc.start_latitude, tmc.start_longitude)
    end = point_key(tmc.end_latitude, tmc.end_longitude)
    head = tmc.geometry.map(bearing)

    def step(link, before):
        key = start[link] if before else end[link]
        cand = [c for c in (end[end == key].index if before else start[start == key].index) if c != link]
        return min(cand, key=lambda c: turn(head[c], head[link])) if cand else None

    chain, total, link = [target], 0.0, target
    while total < upstream_mi and (link := step(link, True)) is not None and link not in chain:
        chain.insert(0, link)
        total += float(tmc.loc[link, "miles"])
    total, link = 0.0, target
    while total < downstream_mi and (link := step(link, False)) is not None and link not in chain:
        chain.append(link)
        total += float(tmc.loc[link, "miles"])
    return chain


def business_slots(first, last):
    days = pd.bdate_range(first, last)
    offsets = pd.to_timedelta(np.arange(START_MIN, END_MIN, STEP_MIN), unit="min")
    return pd.DatetimeIndex([d + o for d in days for o in offsets]), set(days.date)


def local_time(s):
    return pd.to_datetime(s.astype(str).str.replace(r"([+-]\d{2}:\d{2}|Z)$", "", regex=True))


def speeds(files, links, interval_min, slots):
    df = pd.concat(pd.read_csv(f, usecols=["tmc_code", "measurement_tstamp", "speed"]) for f in files)
    df = df[df.tmc_code.isin(links)]
    df["end"] = local_time(df.measurement_tstamp) + pd.Timedelta(minutes=interval_min)
    df["bin"] = df.end.dt.ceil(f"{STEP_MIN}min")
    df["inv"] = 1.0 / df.speed.where(df.speed > 0)
    wide = df.groupby(["bin", "tmc_code"]).inv.mean().rdiv(1.0).unstack()
    return wide.reindex(index=slots, columns=links)


def reports(files, shapes, links, days):
    df = pd.concat(pd.read_csv(f) for f in files)
    df = df.dropna(subset=["Start time", "Closed time", "Latitude", "Longitude"])
    df["start_time"], df["end_time"] = local_time(df["Start time"]), local_time(df["Closed time"])
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
    area = shapes[shapes.tmc.isin(links)].copy()
    area["geometry"] = area.to_crs(epsg=3857).buffer(10, resolution=1).to_crs(epsg=4326)
    hit = gpd.sjoin(points, area[["tmc", "geometry"]], how="inner", predicate="within")
    minute = lambda t: t.dt.hour * 60 + t.dt.minute
    inside = lambda t: (minute(t) >= START_MIN) & (minute(t) < END_MIN)
    keep = (hit.start_time.dt.date == hit.end_time.dt.date) & hit.start_time.dt.date.isin(days)
    keep &= inside(hit.start_time) | inside(hit.end_time)
    hit = hit[keep].sort_values(["start_time", "tmc"])
    return pd.DataFrame(dict(report_id=[f"R{i:06d}" for i in range(len(hit))], link_id=hit.tmc.values,
                             start_time=hit.start_time.dt.strftime("%Y-%m-%d %H:%M:%S").values,
                             end_time=hit.end_time.dt.strftime("%Y-%m-%d %H:%M:%S").values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tmc-shape", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--speed", nargs="+", required=True)
    p.add_argument("--speed-interval-min", type=int, default=1)
    p.add_argument("--incidents", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--upstream-mi", type=float, default=2.5)
    p.add_argument("--downstream-mi", type=float, default=1.5)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    shapes = gpd.read_file(a.tmc_shape)
    links = corridor(shapes, a.target, a.upstream_mi, a.downstream_mi)
    length = shapes.set_index("tmc").loc[links, "miles"].astype(float).values
    slots, days = business_slots(a.start, a.end)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(link_id=links, order=range(len(links)), length_mi=length,
                      upstream_link=[""] + links[:-1], downstream_link=links[1:] + [""])).to_csv(out / "links.csv", index=False)
    speed = speeds(a.speed, links, a.speed_interval_min, slots)
    speed.insert(0, "timestamp", slots.strftime("%Y-%m-%d %H:%M"))
    speed.to_csv(out / "speed_5min.csv", index=False, float_format="%.2f")
    reports(a.incidents, shapes, links, days).to_csv(out / "incident_reports.csv", index=False)


if __name__ == "__main__":
    main()
