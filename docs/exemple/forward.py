"""单低阻体 TM 二维 MT 正演和阻抗数据转换。"""

from __future__ import annotations

import numpy as np
from discretize import TensorMesh
from pymatsolver import SolverLU
from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem


VACUUM_PERMEABILITY = 4.0 * np.pi * 1e-7


def station_positions(config: dict) -> np.ndarray:
    """从起止位置及间距生成测线测点。"""

    parameters = config["stations"]
    return np.arange(parameters["start_m"], parameters["stop_m"] + parameters["spacing_m"] / 2.0,
                     parameters["spacing_m"], dtype=float)


def frequencies(config: dict) -> np.ndarray:
    """生成由低到高的对数等距频率。"""

    parameters = config["frequencies"]
    return np.logspace(np.log10(parameters["minimum_hz"]), np.log10(parameters["maximum_hz"]),
                       int(parameters["count"]))


def build_impedance_survey(frequencies_hz: np.ndarray, stations_m: np.ndarray):
    """每个频率按全部测点实部、全部测点虚部排列阻抗。"""

    locations = np.c_[stations_m, np.zeros(stations_m.size)]
    sources = []
    for frequency in frequencies_hz:
        real = nsem.receivers.Impedance(locations, orientation="xy", component="real")
        imag = nsem.receivers.Impedance(locations, orientation="xy", component="imag")
        sources.append(nsem.sources.Planewave([real, imag], float(frequency)))
    return nsem.survey.Survey(sources)


def build_rho_phase_survey(frequencies_hz: np.ndarray, stations_m: np.ndarray):
    """为反演构造每频先视电阻率、后相位的观测排列。"""

    locations = np.c_[stations_m, np.zeros(stations_m.size)]
    sources = []
    for frequency in frequencies_hz:
        rho = nsem.receivers.Impedance(locations, orientation="xy", component="apparent_resistivity")
        phase = nsem.receivers.Impedance(locations, orientation="xy", component="phase")
        sources.append(nsem.sources.Planewave([rho, phase], float(frequency)))
    return nsem.survey.Survey(sources)


def convert_impedance(prediction: np.ndarray, frequencies_hz: np.ndarray, station_count: int) -> dict:
    """按物理公式从复阻抗计算视电阻率及两种相位分支。"""

    n_frequency = frequencies_hz.size
    if prediction.size != 2 * n_frequency * station_count:
        raise ValueError("阻抗数据数量与频率、测点及接收器排列不符")
    blocks = np.asarray(prediction).reshape(n_frequency, 2, station_count)
    raw_impedance = blocks[:, 0, :] + 1j * blocks[:, 1, :]
    angular_frequency = 2.0 * np.pi * frequencies_hz[:, None]
    rho = np.abs(raw_impedance) ** 2 / (VACUUM_PERMEABILITY * angular_frequency)
    raw_phase = np.angle(raw_impedance, deg=True)
    display_phase = np.mod(np.angle(-raw_impedance, deg=True), 180.0)
    if np.any(rho <= 0) or not np.all(np.isfinite(rho)) or not np.all(np.isfinite(raw_phase)):
        raise ValueError("正演阻抗转换得到无效视电阻率或相位")
    return {
        "impedance": raw_impedance,
        "apparent_resistivity": rho,
        "phase_raw_deg": raw_phase,
        "phase_display_deg": display_phase,
    }


def simulate(mesh: TensorMesh, log_conductivity: np.ndarray,
             frequencies_hz: np.ndarray, stations_m: np.ndarray) -> dict:
    """调用项目约定的 TM 二维电场模拟器，不注入静态位移或噪声。"""

    survey = build_impedance_survey(frequencies_hz, stations_m)
    simulation = nsem.simulation.Simulation2DElectricField(
        mesh, survey=survey, sigmaMap=maps.ExpMap(mesh),
        solver=SolverLU, storeInnerProduct=True,
    )
    prediction = np.asarray(simulation.dpred(log_conductivity), dtype=float)
    return {"dpred": prediction, **convert_impedance(prediction, frequencies_hz, stations_m.size)}


def verify_receiver_convention(mesh: TensorMesh, log_conductivity: np.ndarray,
                               frequencies_hz: np.ndarray, stations_m: np.ndarray,
                               derived: dict) -> tuple[np.ndarray, dict]:
    """以 SimPEG 原生接收器核对公式转换与反演相位分支。"""

    survey = build_rho_phase_survey(frequencies_hz, stations_m)
    simulation = nsem.simulation.Simulation2DElectricField(
        mesh, survey=survey, sigmaMap=maps.ExpMap(mesh),
        solver=SolverLU, storeInnerProduct=True,
    )
    direct = np.asarray(simulation.dpred(log_conductivity), dtype=float)
    direct_blocks = direct.reshape(frequencies_hz.size, 2, stations_m.size)
    rho_difference = np.max(np.abs(direct_blocks[:, 0, :] / derived["apparent_resistivity"] - 1.0))
    phase_difference = np.max(np.abs(direct_blocks[:, 1, :] - derived["phase_raw_deg"]))
    if rho_difference > 1e-8 or phase_difference > 1e-8:
        raise AssertionError("阻抗公式与反演接收器未使用同一视电阻率或相位约定")
    return direct, {"rho_relative_difference_max": float(rho_difference),
                    "phase_difference_max_deg": float(phase_difference)}
