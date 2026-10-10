"""构建 TE、TM 模式的阻抗接收器与频率源。"""

import numpy as np
from simpeg.electromagnetics import natural_source as nsem

from .settings import FREQUENCIES_HZ, STATION_X_M


def build_survey(mode: str) -> nsem.survey.Survey:
    """每频先排列全部测点实部，再排列全部测点虚部。"""
    if mode not in ("TE", "TM"):
        raise ValueError("mode 只能是 TE 或 TM")
    locations = np.column_stack((STATION_X_M, np.zeros_like(STATION_X_M)))
    # x 为剖面方向；沿走向电场为 TE，跨走向电场为 TM。
    orientation = "yx" if mode == "TE" else "xy"
    sources = []
    for frequency in FREQUENCIES_HZ:
        real_receiver = nsem.receivers.Impedance(locations, orientation=orientation, component="real")
        imag_receiver = nsem.receivers.Impedance(locations, orientation=orientation, component="imag")
        sources.append(nsem.sources.Planewave([real_receiver, imag_receiver], frequency))
    return nsem.survey.Survey(sources)
