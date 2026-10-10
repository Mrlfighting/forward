"""旧 L4 观测分量顺序、三方法接收器与误差向量。"""

from __future__ import annotations

import numpy as np
from simpeg.electromagnetics import natural_source as nsem


def _locations(stations: np.ndarray) -> np.ndarray:
    return np.c_[stations, np.zeros(stations.size)]


def full_survey(frequencies: np.ndarray, stations: np.ndarray) -> nsem.survey.Survey:
    locations = _locations(stations)
    sources = []
    for frequency in frequencies:
        receivers = [
            nsem.receivers.Impedance(
                locations, orientation="xy", component="apparent_resistivity"
            ),
            nsem.receivers.Impedance(locations, orientation="xy", component="phase"),
        ]
        sources.append(nsem.sources.Planewave(receivers, frequency))
    return nsem.survey.Survey(sources)


def phase_corrected_survey(frequencies: np.ndarray, stations: np.ndarray,
                           static_indices: np.ndarray) -> nsem.survey.Survey:
    normal = np.setdiff1d(np.arange(stations.size), static_indices)
    normal_locations = _locations(stations[normal])
    all_locations = _locations(stations)
    sources = []
    for frequency in frequencies:
        receivers = [
            nsem.receivers.Impedance(
                normal_locations, orientation="xy", component="apparent_resistivity"
            ),
            nsem.receivers.Impedance(all_locations, orientation="xy", component="phase"),
        ]
        sources.append(nsem.sources.Planewave(receivers, frequency))
    return nsem.survey.Survey(sources)


def transformed_survey(frequencies: np.ndarray, stations: np.ndarray,
                       static_indices: np.ndarray) -> nsem.survey.Survey:
    normal = np.setdiff1d(np.arange(stations.size), static_indices)
    normal_locations = _locations(stations[normal])
    all_locations = _locations(stations)
    static_locations = _locations(stations[static_indices])
    sources = []
    for index, frequency in enumerate(frequencies):
        receivers = [
            nsem.receivers.Impedance(
                normal_locations, orientation="xy", component="apparent_resistivity"
            ),
            nsem.receivers.Impedance(all_locations, orientation="xy", component="phase"),
        ]
        if index < frequencies.size - 1:
            receivers.append(nsem.receivers.Impedance(
                static_locations, orientation="xy", component="apparent_resistivity"
            ))
        sources.append(nsem.sources.Planewave(receivers, frequency))
    return nsem.survey.Survey(sources)


def join_full(rho_a: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return np.stack((rho_a, phase), axis=1).reshape(-1)


def select_regular(rho_a: np.ndarray, phase: np.ndarray,
                   normal_indices: np.ndarray) -> np.ndarray:
    return np.concatenate((rho_a[:, normal_indices], phase), axis=1).ravel()


def full_standard_deviation(rho_a: np.ndarray, phase: np.ndarray,
                            rho_relative_error: float, phase_error_deg: float) -> np.ndarray:
    rho_std = np.maximum(rho_relative_error * np.abs(rho_a), 1.0e-12)
    phase_std = np.full_like(phase, phase_error_deg)
    return join_full(rho_std, phase_std)
