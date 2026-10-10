"""绘制 L4 识别诊断、三方法模型与逐点观测拟合。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from discretize import TensorMesh
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mt_static.realdata.l4.data import normal_phase


def model_grid(mesh: TensorMesh, values: np.ndarray, x_km: np.ndarray,
               z_km: np.ndarray) -> np.ndarray:
    """仅对显示网格插值和平滑，保存的反演模型不变。"""

    grid = values.reshape(mesh.shape_cells, order="F")
    interpolator = RegularGridInterpolator(
        (mesh.cell_centers_x, mesh.cell_centers_y), grid,
        method="linear", bounds_error=False, fill_value=None,
    )
    xx, zz = np.meshgrid(x_km * 1000.0, z_km * 1000.0)
    image = interpolator(np.c_[xx.ravel(), zz.ravel()]).reshape(xx.shape)
    return gaussian_filter(image, sigma=(1.2, 1.4), mode="nearest")


def plot_identification(destination: Path, stations: np.ndarray,
                        frequencies: np.ndarray, votes: np.ndarray,
                        selected: np.ndarray) -> None:
    counts = votes.sum(axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    colors = ["tab:red" if index in selected else "tab:blue" for index in range(stations.size)]
    axes[0].bar(np.arange(stations.size), counts, color=colors, alpha=0.82)
    axes[0].set_xticks(np.arange(stations.size), [f"S{index + 1:02d}" for index in range(stations.size)])
    axes[0].set_ylabel("出现候选的频率数")
    axes[0].set_title("L4｜正式小波识别器的逐频候选；红色为人工确认的反演测点")
    y_edges = np.r_[frequencies[0] / np.sqrt(frequencies[1] / frequencies[0]),
                    np.sqrt(frequencies[:-1] * frequencies[1:]),
                    frequencies[-1] * np.sqrt(frequencies[-1] / frequencies[-2])]
    x_edges = np.r_[stations[0] - 250.0, 0.5 * (stations[:-1] + stations[1:]),
                    stations[-1] + 250.0] / 1000.0
    image = axes[1].pcolormesh(x_edges, y_edges, votes.astype(int), cmap="Greys", vmin=0, vmax=1)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("测点位置（千米）")
    axes[1].set_ylabel("频率（Hz，高频在上）")
    axes[1].set_title("逐频候选矩阵；黑色表示该频率在该测点出现候选")
    for index in selected:
        axes[1].axvline(stations[index] / 1000.0, color="tab:red", linewidth=1.0, alpha=0.65)
    fig.colorbar(image, ax=axes[1], ticks=[0, 1], label="候选标记")
    fig.savefig(destination, dpi=220, facecolor="white")
    plt.close(fig)


def plot_models(destination: Path, mesh: TensorMesh, models: dict[str, np.ndarray],
                stations: np.ndarray, selected: np.ndarray, depth_km: float) -> dict:
    x_km = np.linspace(stations.min() / 1000.0 - 0.25,
                       stations.max() / 1000.0 + 0.25, 700)
    z_km = np.linspace(-depth_km, 0.0, 500)
    rendered = {name: model_grid(mesh, values, x_km, z_km) for name, values in models.items()}
    pooled = np.concatenate([values.ravel() for values in rendered.values()])
    low, high = np.quantile(pooled, [0.01, 0.99])
    low = np.floor(float(low) * 4.0) / 4.0
    high = np.ceil(float(high) * 4.0) / 4.0
    if high - low < 1.0:
        mid = 0.5 * (low + high)
        low, high = mid - 0.5, mid + 0.5
    labels = {
        "full": "（a）全部测点：视电阻率＋相位",
        "phase": "（b）普通点完整数据；候选点仅相位",
        "shape": "（c）相位保留＋候选点视电阻率形态",
    }
    fig, axes = plt.subplots(3, 1, figsize=(11.4, 12.6), sharex=True, layout="constrained")
    xx, zz = np.meshgrid(x_km, z_km)
    image = None
    for axis, name in zip(axes, ("full", "phase", "shape")):
        image = axis.contourf(xx, zz, rendered[name], levels=np.linspace(low, high, 61),
                              cmap="turbo", extend="both")
        axis.scatter(stations / 1000.0, np.full(stations.size, -0.003 * depth_km),
                     marker="v", s=25, color="black", zorder=6)
        axis.scatter(stations[selected] / 1000.0, np.full(selected.size, -0.003 * depth_km),
                     marker="v", s=45, color="red", zorder=7)
        axis.set_xlim(x_km[0], x_km[-1])
        axis.set_ylim(-depth_km, 0.0)
        axis.set_ylabel("深度（千米）")
        axis.set_title(labels[name])
    axes[-1].set_xlabel("测线位置（千米）")
    assert image is not None
    fig.colorbar(image, ax=axes.tolist(), fraction=0.032, pad=0.025,
                 label="lg［电阻率／(Ω·m)］")
    fig.savefig(destination, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"depth_km": depth_km, "color_limits_log10_ohm_m": [float(low), float(high)]}


def plot_difference(destination: Path, mesh: TensorMesh, phase: np.ndarray,
                    shape: np.ndarray, stations: np.ndarray, selected: np.ndarray) -> None:
    x_km = np.linspace(stations.min() / 1000.0 - 0.25,
                       stations.max() / 1000.0 + 0.25, 700)
    z_km = np.linspace(-20.0, 0.0, 500)
    values = model_grid(mesh, shape - phase, x_km, z_km)
    bound = max(0.1, float(np.quantile(np.abs(values), 0.99)))
    xx, zz = np.meshgrid(x_km, z_km)
    fig, axis = plt.subplots(figsize=(11, 4.5), layout="constrained")
    image = axis.contourf(xx, zz, values, levels=np.linspace(-bound, bound, 61),
                          cmap="RdBu_r", extend="both")
    axis.scatter(stations / 1000.0, np.full(stations.size, -0.06),
                 marker="v", s=25, color="black")
    axis.scatter(stations[selected] / 1000.0, np.full(selected.size, -0.06),
                 marker="v", s=45, color="red")
    axis.set(xlim=(x_km[0], x_km[-1]), ylim=(-20.0, 0.0),
             xlabel="测线位置（千米）", ylabel="深度（千米）",
             title="形态约束减相位保留：lg 电阻率差异")
    fig.colorbar(image, ax=axis, label="lg 电阻率差值")
    fig.savefig(destination, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fits(destination: Path, prepared, models: dict[str, dict[str, np.ndarray]],
              selected: np.ndarray) -> dict:
    frequencies = prepared["frequencies_hz"]
    stations = prepared["stations_m"]
    observed_rho = prepared["rho_xy_ohm_m"]
    observed_phase = prepared["phase_normal_deg"]
    predictions = {}
    for name, archive in models.items():
        full = archive["predicted_full"].reshape(frequencies.size, 2, stations.size)
        predictions[name] = (full[:, 0, :], normal_phase(full[:, 1, :]))
    colors = {"full": "tab:blue", "phase": "tab:orange", "shape": "tab:green"}
    labels = {"full": "全数据", "phase": "相位保留", "shape": "形态约束"}
    destination.mkdir(parents=True, exist_ok=True)
    metrics = []
    with PdfPages(destination / "L4_全部测点拟合.pdf") as pdf:
        for start in range(0, stations.size, 4):
            stop = min(start + 4, stations.size)
            fig, axes = plt.subplots(stop - start, 2, figsize=(12, 3.25 * (stop - start)),
                                     layout="constrained", squeeze=False)
            for row, index in enumerate(range(start, stop)):
                rho_axis, phase_axis = axes[row]
                rho_axis.plot(frequencies, observed_rho[:, index], "ko", markersize=2.5,
                              label="观测")
                phase_axis.plot(frequencies, observed_phase[:, index], "ks", markersize=2.5,
                                label="观测")
                for name in ("full", "phase", "shape"):
                    rho_pred, phase_pred = predictions[name]
                    rho_axis.plot(frequencies, rho_pred[:, index], color=colors[name],
                                  linewidth=1.3, label=labels[name])
                    phase_axis.plot(frequencies, phase_pred[:, index], color=colors[name],
                                    linewidth=1.3, label=labels[name])
                for axis in (rho_axis, phase_axis):
                    axis.set_xscale("log")
                    axis.set_xlim(float(frequencies.max()), float(frequencies.min()))
                    axis.grid(True, alpha=0.3)
                    axis.set_xlabel("频率（Hz，高频在左）")
                rho_axis.set_yscale("log")
                rho_axis.set_ylabel("视电阻率（Ω·m）")
                phase_axis.set_ylabel("相位（°）")
                label = f"S{index + 1:02d}｜{stations[index]:.0f} m"
                if index in selected:
                    label += "｜候选静态点"
                rho_axis.set_title(label + "｜视电阻率")
                phase_axis.set_title(label + "｜相位")
                if row == 0:
                    rho_axis.legend(fontsize=8, ncol=4)
                    phase_axis.legend(fontsize=8, ncol=4)
                row_metrics = {"station_id": f"S{index + 1:02d}",
                               "station_x_m": float(stations[index]),
                               "selected_static": bool(index in selected)}
                for name, (rho_pred, phase_pred) in predictions.items():
                    row_metrics[f"{name}_rho_relative_rms"] = float(np.sqrt(np.mean(
                        ((rho_pred[:, index] - observed_rho[:, index]) / observed_rho[:, index]) ** 2
                    )))
                    row_metrics[f"{name}_phase_rms_deg"] = float(np.sqrt(np.mean(
                        (phase_pred[:, index] - observed_phase[:, index]) ** 2
                    )))
                metrics.append(row_metrics)
            page = start // 4 + 1
            fig.savefig(destination / f"L4_测点拟合_{page:02d}.png", dpi=200,
                        bbox_inches="tight", facecolor="white")
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)
    with (destination / "L4_逐点拟合指标.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    return {"station_count": len(metrics), "page_count": (len(metrics) + 3) // 4}


def main() -> None:
    prepared = np.load(ROOT / "data/realdata/l4/prepared.npz")
    result_root = ROOT / "results/realdata/l4"
    summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
    if set(summary["methods"]) != {"full", "phase", "shape"}:
        raise RuntimeError("三方法结果尚未齐全，不能绘制最终对比图")
    setup = json.loads((result_root / "setup.json").read_text(encoding="utf-8"))
    selected = np.asarray(setup["selected_static_indices"], dtype=int)
    detection = np.load(result_root / "identification/detection_diagnostics.npz")
    mesh_data = np.load(result_root / "inversion/mesh.npz")
    mesh = TensorMesh([mesh_data["hx"], mesh_data["hz"]], x0=mesh_data["x0"])
    active = mesh_data["active_cells"].astype(bool)
    models = {name: np.load(result_root / f"inversion/{name}.npz")
              for name in ("full", "phase", "shape")}
    log_rho = {}
    for name, archive in models.items():
        values = -archive["recovered_full"] / np.log(10.0)
        values[~active] = np.log10(setup["inversion_parameters"]["starting_rho"])
        log_rho[name] = values
    figures = result_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_identification(figures / "L4_静态候选逐频诊断.png",
                        prepared["stations_m"], prepared["frequencies_hz"],
                        detection["per_frequency_votes"], selected)
    plot_summary = {"display_only_smoothing": "线性插值＋高斯 sigma=(1.2,1.4) 像素"}
    for name, depth in (("浅部20千米", 20.0), ("全深度300千米", 300.0)):
        plot_summary[name] = plot_models(
            figures / f"L4_三方法反演对比_{name}.png", mesh, log_rho,
            prepared["stations_m"], selected, depth,
        )
    plot_difference(figures / "L4_形态减相位保留差异.png", mesh,
                    log_rho["phase"], log_rho["shape"],
                    prepared["stations_m"], selected)
    plot_summary["fits"] = plot_fits(figures, prepared, models, selected)
    (figures / "plot_summary.json").write_text(
        json.dumps(plot_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("L4 识别诊断、三方法模型和逐点拟合图已保存", flush=True)


if __name__ == "__main__":
    main()
