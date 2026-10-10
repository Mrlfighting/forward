"""执行两种极化、两套模型的二维 MT 物理正演。"""

import numpy as np
from discretize import TensorMesh
from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem

from .response import convert_response
from .survey import build_survey


def simulate_mode(
    mesh: TensorMesh,
    models: dict[str, np.ndarray],
    mode: str,
    baseline_dpred: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """同一模式下共用测量设置，必要时复用已核对的基线阻抗。"""
    survey = build_survey(mode)
    simulation_type = (
        nsem.simulation.Simulation2DMagneticField
        if mode == "TE"
        else nsem.simulation.Simulation2DElectricField
    )
    simulation = simulation_type(mesh, survey=survey, sigmaMap=maps.ExpMap(mesh), storeInnerProduct=True)
    responses = {}
    for model_name, log_conductivity in models.items():
        if model_name == "without_shallow" and baseline_dpred is not None:
            print(f"{mode}：复用已核对的无浅部体基线……", flush=True)
            dpred = np.asarray(baseline_dpred, dtype=float).copy()
        else:
            print(f"{mode}：计算{model_name}……", flush=True)
            dpred = np.asarray(simulation.dpred(log_conductivity), dtype=float)
        responses[model_name] = {"dpred": dpred, **convert_response(dpred)}
    return responses
