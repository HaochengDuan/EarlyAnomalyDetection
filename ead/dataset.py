from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .timegrid import N_SLOTS, SLOT_MINUTES, SLOT_START_MINUTE


@dataclass
class Corridor:
    link_ids: list
    lengths: np.ndarray

    def index(self, link_id):
        return self.link_ids.index(link_id)

    def within(self, link_id, direction, range_mi):
        i = self.index(link_id)
        step = -1 if direction == "upstream" else 1
        out, gap, j = [], 0.0, i + step
        while 0 <= j < len(self.link_ids) and gap < range_mi:
            out.append(self.link_ids[j])
            gap += self.lengths[j]
            j += step
        return out

    def subgraph(self, link_id, upstream_mi, downstream_mi):
        return [link_id] + self.within(link_id, "upstream", upstream_mi) + self.within(link_id, "downstream", downstream_mi)


@dataclass
class LinkData:
    dates: pd.DatetimeIndex
    corridor: Corridor
    speed: np.ndarray
    reports: pd.DataFrame


def load(folder):
    folder = Path(folder)
    links = pd.read_csv(folder / "links.csv").sort_values("order")
    corridor = Corridor(list(links.link_id), links.length_mi.to_numpy(float))

    df = pd.read_csv(folder / "speed_5min.csv")
    ts = pd.to_datetime(df.pop("timestamp"))
    dates = pd.DatetimeIndex(ts.dt.normalize().unique())
    expected = (np.repeat(dates.values, N_SLOTS)
                + pd.to_timedelta(np.tile(SLOT_START_MINUTE + SLOT_MINUTES * np.arange(N_SLOTS), len(dates)), unit="min").values)
    if len(ts) != len(expected) or not np.array_equal(ts.values, expected):
        raise ValueError("speed_5min.csv must contain every 5-min slot from 05:30 to 20:55 for each day, in order")
    speed = df[corridor.link_ids].to_numpy(float).reshape(len(dates), N_SLOTS, len(corridor.link_ids))

    reports = pd.read_csv(folder / "incident_reports.csv", parse_dates=["start_time", "end_time"])
    return LinkData(dates, corridor, speed, reports)
