import numpy as np

SLOT_START_MINUTE = 330
SLOT_MINUTES = 5
N_SLOTS = 186


def slot_minutes():
    return SLOT_START_MINUTE + SLOT_MINUTES * np.arange(N_SLOTS)
