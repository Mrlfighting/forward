"""读取并校验论文 2.1 同一模型族的场景配置。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .settings import FREQUENCIES_HZ, STATION_X_M


IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


def _check_identifier(value: object, field: str) -> str:
    """限制标识符，避免场景路径逃出指定结果目录。"""

    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} 只能使用小写英文字母、数字和下划线，且以字母开头")
    return value


def _check_body(body: dict, field: str) -> dict:
    """检查单个异常体的电阻率、坐标与编号。"""

    required = {"body_id", "x_min_m", "x_max_m", "z_min_m", "z_max_m", "resistivity_ohm_m"}
    if set(body) != required:
        raise ValueError(f"{field} 字段应为 {sorted(required)}")
    checked = {"body_id": _check_identifier(body["body_id"], f"{field}.body_id")}
    for name in required - {"body_id"}:
        value = float(body[name])
        if not np.isfinite(value):
            raise ValueError(f"{field}.{name} 必须有限")
        checked[name] = value
    if checked["x_min_m"] >= checked["x_max_m"] or checked["z_min_m"] >= checked["z_max_m"]:
        raise ValueError(f"{field} 的坐标下界必须小于上界")
    if checked["resistivity_ohm_m"] <= 0:
        raise ValueError(f"{field} 的电阻率必须为正")
    if checked["z_max_m"] > 0:
        raise ValueError(f"{field} 不得超出地表")
    return checked


def load_base_config(path: Path) -> dict:
    """校验公共模型与当前已验证的网格、测点和频率版本一致。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("base_id") != "section21_base_v1" or payload.get("mesh_id") != "section21_mesh_v1":
        raise ValueError("当前场景入口仅支持已验证的 section21_base_v1 网格模型族")
    background = float(payload["background_resistivity_ohm_m"])
    if not np.isfinite(background) or background <= 0:
        raise ValueError("背景电阻率必须为正且有限")
    bodies = [_check_body(body, "deep_bodies") for body in payload["deep_bodies"]]
    if len(bodies) != 2 or len({body["body_id"] for body in bodies}) != 2:
        raise ValueError("当前基础模型应包含两个编号不同的深部异常体")
    stations = payload["stations"]
    if float(stations["step_m"]) <= 0 or float(stations["stop_m"]) < float(stations["start_m"]):
        raise ValueError("测点间距须为正，且终点不得小于起点")
    station_x = np.arange(
        float(stations["start_m"]),
        float(stations["stop_m"]) + float(stations["step_m"]) / 2.0,
        float(stations["step_m"]),
    )
    frequency = payload["frequencies"]
    if int(frequency["count"]) < 1:
        raise ValueError("频率数量须为正")
    frequencies = np.logspace(
        float(frequency["log10_min_hz"]),
        float(frequency["log10_max_hz"]),
        int(frequency["count"]),
    )
    if not np.array_equal(station_x, STATION_X_M) or not np.array_equal(frequencies, FREQUENCIES_HZ):
        raise ValueError("公共配置的测点或频率与当前已验证的测量版本不一致")
    return {**payload, "background_resistivity_ohm_m": background, "deep_bodies": bodies}


def load_scenario_config(path: Path, base_id: str) -> dict:
    """读取任意数量的浅部体与需要绘制曲线的测点。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    scenario_id = _check_identifier(payload.get("scenario_id"), "scenario_id")
    if path.stem != scenario_id:
        raise ValueError("场景文件名必须与 scenario_id 相同")
    if payload.get("base_id") != base_id:
        raise ValueError("场景引用的公共模型版本不匹配")
    bodies = [_check_body(body, "shallow_bodies") for body in payload["shallow_bodies"]]
    ids = [body["body_id"] for body in bodies]
    if len(ids) != len(set(ids)):
        raise ValueError("浅部体编号不得重复")
    if any(body["z_max_m"] != 0.0 for body in bodies):
        raise ValueError("当前浅部体必须接触地表，即 z_max_m=0")
    positions = [float(value) for value in payload["curve_station_x_m"]]
    if not positions or len(positions) != len(set(positions)):
        raise ValueError("曲线测点列表不得为空或重复")
    if any(not np.any(np.isclose(STATION_X_M, value, atol=1e-9)) for value in positions):
        raise ValueError("曲线测点必须是当前实际测点")
    return {**payload, "shallow_bodies": bodies, "curve_station_x_m": positions}
