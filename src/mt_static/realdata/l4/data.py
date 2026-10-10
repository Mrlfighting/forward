"""按旧 L4 流程准备 XY 分量观测，不改变原始频点。"""

from __future__ import annotations

import hashlib
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = ("Frequency", "MT_Rxy", "MT_Pxy")


@dataclass(frozen=True)
class PreparedL4:
    station_codes: np.ndarray
    stations_m: np.ndarray
    frequencies_hz: np.ndarray
    rho_xy_ohm_m: np.ndarray
    phase_raw_deg: np.ndarray
    phase_normal_deg: np.ndarray
    phase_simpeg_deg: np.ndarray
    full_observed: np.ndarray
    source_sha256: dict[str, str]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normal_phase(phase_deg: np.ndarray) -> np.ndarray:
    """保留旧程序的 180 度折叠与相位分支约定。"""

    wrapped = np.mod(np.asarray(phase_deg, dtype=float), 180.0)
    return np.minimum(wrapped, 180.0 - wrapped)


def prepare_l4(raw_dir: Path, minimum_frequency_hz: float = 0.00114,
               station_spacing_m: float = 500.0) -> PreparedL4:
    """读取 15 个 CSV，并要求每个测点有完全相同的原始频率。"""

    if minimum_frequency_hz <= 0 or station_spacing_m <= 0:
        raise ValueError("频率下限和测点间距必须为正")
    files = sorted(raw_dir.glob("*.csv"), key=lambda path: float(path.stem))
    if not files:
        raise FileNotFoundError(f"没有找到 L4 CSV：{raw_dir}")
    station_codes = np.asarray([float(path.stem) for path in files])
    if np.unique(station_codes).size != station_codes.size:
        raise ValueError("测点文件名重复")
    stations = np.arange(len(files), dtype=float) * station_spacing_m
    frequencies = None
    rho_columns = []
    phase_columns = []
    hashes = {}
    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"{path.name} 缺少表头")
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"{path.name} 缺少字段：{missing}")
            rows = []
            for row in reader:
                try:
                    values = tuple(float(row[column]) for column in REQUIRED_COLUMNS)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path.name} 含非数值观测") from exc
                if values[0] >= minimum_frequency_hz:
                    rows.append(values)
        values = np.asarray(sorted(rows, key=lambda item: item[0]), dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"{path.name} 没有有效频率")
        if np.unique(values[:, 0]).size != values.shape[0]:
            raise ValueError(f"{path.name} 选取频率存在重复")
        selected = values[:, 0]
        if frequencies is None:
            frequencies = selected
        elif not np.array_equal(frequencies, selected):
            raise ValueError(f"{path.name} 的频率与其他测点不同；本流程不插值")
        if not np.isfinite(values).all() or np.any(values[:, 1] <= 0):
            raise ValueError(f"{path.name} 含无效频率、视电阻率或相位")
        rho_columns.append(values[:, 1])
        phase_columns.append(values[:, 2])
        hashes[path.name] = file_sha256(path)
    assert frequencies is not None
    rho = np.column_stack(rho_columns)
    phase_raw = np.column_stack(phase_columns)
    phase_normal = normal_phase(phase_raw)
    phase_simpeg = phase_normal - 180.0
    full = np.stack((rho, phase_simpeg), axis=1).reshape(-1)
    return PreparedL4(station_codes, stations, frequencies, rho, phase_raw,
                      phase_normal, phase_simpeg, full, hashes)


def save_prepared(path: Path, prepared: PreparedL4) -> None:
    """将反演必需的数组集中保存，保留频率、分量、测点顺序。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        station_codes=prepared.station_codes,
        stations_m=prepared.stations_m,
        frequencies_hz=prepared.frequencies_hz,
        rho_xy_ohm_m=prepared.rho_xy_ohm_m,
        phase_raw_deg=prepared.phase_raw_deg,
        phase_normal_deg=prepared.phase_normal_deg,
        phase_simpeg_deg=prepared.phase_simpeg_deg,
        full_observed=prepared.full_observed,
    )
