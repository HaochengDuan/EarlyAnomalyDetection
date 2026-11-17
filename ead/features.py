import numpy as np
import pandas as pd

from .timegrid import N_SLOTS, SLOT_MINUTES, SLOT_START_MINUTE


def free_flow_speed(speed):
    return np.nanpercentile(speed.reshape(-1, speed.shape[-1]), 85, axis=0)


def impute(speed):
    return np.where(np.isnan(speed), free_flow_speed(speed)[None, None, :], speed)


def slowdown_speed(speed, corridor, range_mi=0.3):
    sd = np.zeros_like(speed)
    for i, link in enumerate(corridor.link_ids):
        up = [corridor.index(u) for u in corridor.within(link, "upstream", range_mi)]
        if up:
            sd[..., i] = np.maximum(speed[..., up].mean(axis=-1) - speed[..., i], 0.0)
    return sd


def travel_time_index(speed):
    return np.maximum(free_flow_speed(speed)[None, None, :] / speed, 1.0)


def report_matrix(reports, dates, corridor):
    out = np.zeros((len(dates), N_SLOTS, len(corridor.link_ids)), dtype=np.int8)
    day_index = {d: i for i, d in enumerate(dates)}
    end_of_hours = SLOT_START_MINUTE + SLOT_MINUTES * N_SLOTS
    for r in reports.itertuples(index=False):
        if r.link_id not in corridor.link_ids or r.start_time.normalize() != r.end_time.normalize():
            continue
        d = day_index.get(r.start_time.normalize())
        if d is None:
            continue
        s_min = r.start_time.hour * 60 + r.start_time.minute + r.start_time.second / 60
        e_min = r.end_time.hour * 60 + r.end_time.minute + r.end_time.second / 60
        if not (SLOT_START_MINUTE <= s_min < end_of_hours or SLOT_START_MINUTE <= e_min < end_of_hours):
            continue
        first = max(int(np.ceil((s_min - SLOT_START_MINUTE) / SLOT_MINUTES)), 0)
        last = min(int(np.ceil((e_min - SLOT_START_MINUTE) / SLOT_MINUTES)), N_SLOTS - 1)
        if first > last:
            continue
        out[d, first:last + 1, corridor.index(r.link_id)] = 1
    return out


def seasonal_recurrent_speed(speed, reports, weeks=3):
    days = speed.shape[0]
    num = np.zeros_like(speed)
    den = np.zeros_like(speed)
    for j in range(1, weeks + 1):
        lag = 5 * j
        if lag >= days:
            break
        ok = reports[:-lag] == 0
        num[lag:] += np.where(ok, speed[:-lag], 0.0)
        den[lag:] += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def compute(data, slowdown_range_mi=0.3):
    speed = impute(data.speed)
    report = report_matrix(data.reports, data.dates, data.corridor)
    srs = seasonal_recurrent_speed(speed, report)
    profile = np.nanmean(np.where(report == 0, speed, np.nan), axis=0, keepdims=True)
    return dict(
        speed=speed,
        slowdown=slowdown_speed(speed, data.corridor, slowdown_range_mi),
        tti=travel_time_index(speed),
        srs=np.where(np.isnan(srs), profile, srs),
        report=report,
    )
