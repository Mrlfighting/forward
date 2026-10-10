"""按复杂层状组合模型的真值区域评价二维恢复模型。"""

from __future__ import annotations

import numpy as np
from discretize import TensorMesh

from mt_static.forward.layered_model import layered_background, target_masks


def layered_model_metrics(
    mesh: TensorMesh,
    recovered_log_conductivity: np.ndarray,
    layers: list[dict],
    targets: list[dict],
    background_resistivity: float,
) -> dict:
    """按旧图的目标、层和全区域定义计算可对照指标。"""

    true_model = layered_background(mesh, layers, background_resistivity)
    masks = target_masks(mesh, targets)
    for mask, item in zip(masks, targets):
        true_model[mask] = np.log(1.0 / item["resistivity_ohm_m"])
    centers = mesh.cell_centers
    roi = (np.abs(centers[:, 0]) <= 10000.0) & (centers[:, 1] >= -12000.0) & (centers[:, 1] <= 0.0)
    target_union = np.logical_or.reduce(masks)
    true_rho = np.exp(-true_model)
    recovered_rho = np.exp(-recovered_log_conductivity)

    def rmse(mask: np.ndarray) -> float:
        delta = np.log10(recovered_rho[mask]) - np.log10(true_rho[mask])
        return float(np.sqrt(np.mean(delta**2)))

    def geometric_mean(mask: np.ndarray) -> float:
        return float(np.exp(np.mean(np.log(recovered_rho[mask]))))

    result = {
        "roi_log10_rmse": rmse(roi),
        "layered_background_log10_rmse": rmse(roi & ~target_union),
        "targets_log10_rmse": rmse(target_union),
        "targets_geometric_mean_rho": geometric_mean(target_union),
        "left_target_geometric_mean_rho": geometric_mean(masks[0]),
        "right_target_geometric_mean_rho": geometric_mean(masks[1]),
        "left_target_log10_rmse": rmse(masks[0]),
        "right_target_log10_rmse": rmse(masks[1]),
    }
    for index, item in enumerate(layers, start=1):
        mask = (
            (centers[:, 1] <= -item["top_depth_m"])
            & (centers[:, 1] >= -item["bottom_depth_m"])
            & (np.abs(centers[:, 0]) <= 10000.0)
        )
        result[f"layer_{index}_geometric_mean_rho"] = geometric_mean(mask)
        result[f"layer_{index}_log10_rmse"] = rmse(mask)
    return result
