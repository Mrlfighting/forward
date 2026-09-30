"""
Model Design: MT-only Static Shift Inversion
Background 100 Ohm-m, 4 block anomalies (deep 6km + near-surface static shift)
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from simpeg import maps, utils
from simpeg.utils import model_builder, BreakingChangeWarning
from build_mesh import mesh

warnings.filterwarnings(action='ignore', category=BreakingChangeWarning)

# =============================================================================
# 1. Background Model
# =============================================================================
sig_back = 1.0 / 100.0    # 100 Ohm-m → sigma = 0.01 S/m
sig_low  = 1.0 / 10.0     #  10 Ohm-m → sigma = 0.1  S/m  (red, conductive)
sig_high = 1.0 / 1000.0   # 1000 Ohm-m → sigma = 0.001 S/m (blue, resistive)

m0 = np.log(sig_back) * np.ones(mesh.n_cells)
m_true = m0.copy()

# =============================================================================
# 2. Anomaly Definition
# =============================================================================
# --- 2.1 Left-deep conductive body (red, 10 Ohm-m, 6km x 6km, top at 2km) ---
ind_left_deep = model_builder.get_indices_block(
    np.r_[-8000, -8000],   # p0: x=-8000, z=-8000
    np.r_[-2000, -2000],   # p1: x=-2000, z=-2000
    mesh.cell_centers,
)
m_true[ind_left_deep] = np.log(sig_low)

# --- 2.2 Left-shallow resistive body (blue, 1000 Ohm-m, 4m x 4m, surface) ---
ind_left_shallow = model_builder.get_indices_block(
    np.r_[-5201, -4],      # p0: +1m fp margin
    np.r_[-4899, 0],       # p1: z=0 (surface)
    mesh.cell_centers,
)
m_true[ind_left_shallow] = np.log(sig_high)

# --- 2.3 Right-deep resistive body (blue, 1000 Ohm-m, 6km x 6km, top at 2km) ---
ind_right_deep = model_builder.get_indices_block(
    np.r_[2000, -8000],    # p0: x=2000, z=-8000
    np.r_[8000, -2000],    # p1: x=8000, z=-2000
    mesh.cell_centers,
)
m_true[ind_right_deep] = np.log(sig_high)

# --- 2.4 Right-shallow conductive body (red, 10 Ohm-m, 4m x 4m, surface) ---
ind_right_shallow = model_builder.get_indices_block(
    np.r_[4899, -4],       # p0: -1m fp margin
    np.r_[5201, 0],        # p1: z=0 (surface)
    mesh.cell_centers,
)
m_true[ind_right_shallow] = np.log(sig_low)

# =============================================================================
# 3. Model Info Output
# =============================================================================
print("=" * 60)
print("Model Design: MT Static Shift Inversion")
print("=" * 60)
print(f"  Background:         100 Ohm-m  (sigma = {sig_back:.4f} S/m)")
print(f"  Left-deep (cond):   10 Ohm-m,   {ind_left_deep.size:,} cells (6km x 6km)")
print(f"  Left-shallow (res): 1000 Ohm-m, {ind_left_shallow.size:,} cells (4m x 4m)")
print(f"  Right-deep (res):   1000 Ohm-m,  {ind_right_deep.size:,} cells (6km x 6km)")
print(f"  Right-shallow (cond):10 Ohm-m,   {ind_right_shallow.size:,} cells (4m x 4m)")
print(f"  Total anomaly cells:{ind_left_deep.size + ind_left_shallow.size + ind_right_deep.size + ind_right_shallow.size:,}")
print("=" * 60)

mapping = maps.ExpMap(mesh)

# =============================================================================
# 4. Visualization (only when run directly, not on import)
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # --- Left: Full Depth ---
    out1 = mesh.plot_image(np.exp(m_true), ax=ax1,
                           pcolor_opts={"norm": "log", "cmap": "RdYlBu_r"})
    ax1.set_title("True Model (Full Depth)")
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_ylim(-12000, 500)
    ax1.set_xlim(-10000, 10000)

    for label, xc, zc, color in [
        ("10 Ohm-m", -5000, -5000, 'darkred'),
        ("1k Ohm-m",  5000, -5000, 'darkblue'),
    ]:
        ax1.annotate(label, (xc, zc), textcoords="offset points", xytext=(0, -20),
                     ha='center', fontsize=9, color=color, fontweight='bold')

    plt.colorbar(out1[0], ax=ax1, label="Conductivity (S/m)")

    # --- Right: Near-surface zoom ---
    out2 = mesh.plot_image(np.exp(m_true), ax=ax2,
                           pcolor_opts={"norm": "log", "cmap": "RdYlBu_r"})
    ax2.set_title("True Model (Near-Surface Zoom)")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_ylim(-200, 50)
    ax2.set_xlim(-10000, 10000)

    for label, xc, color in [
        ("1k Ohm-m (surf)", -5000, 'darkblue'),
        ("10 Ohm-m (surf)",  5000, 'darkred'),
    ]:
        ax2.annotate(label, (xc, -5), textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=8, color=color, fontweight='bold')

    plt.colorbar(out2[0], ax=ax2, label="Conductivity (S/m)")
    plt.tight_layout()
    fig.savefig("true_model.png", dpi=150)
    print("\nTrue model figure saved: true_model.png")
    plt.close(fig)

    # =========================================================================
    # 5. 1D Depth Profiles
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    x_tol = 110

    for ax, x_target, title in [
        (ax1, -5000, "Left Column (x ~ -5000 m)"),
        (ax2,  5000, "Right Column (x ~ 5000 m)"),
    ]:
        col_mask = np.abs(mesh.cell_centers[:, 0] - x_target) < x_tol
        col_idx = np.where(col_mask)[0]
        col_z = mesh.cell_centers[col_idx, 1]
        sort_idx = np.argsort(col_z)

        rho_true = 1.0 / np.exp(m_true[col_idx][sort_idx])
        z_sorted = col_z[sort_idx]

        ax.semilogx(rho_true, z_sorted, 'b-', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-2000, color='red', linestyle='--', alpha=0.5, label='Top of deep body (2km)')
        ax.axhline(y=-8000, color='red', linestyle=':', alpha=0.5, label='Bottom of deep body (8km)')
        ax.set_xlabel("Resistivity (Ohm-m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(title)
        ax.set_ylim(-12000, 200)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("true_model_profiles.png", dpi=150)
    print("Depth profiles saved: true_model_profiles.png")
    plt.close(fig)
