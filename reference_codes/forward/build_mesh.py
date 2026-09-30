"""
Mesh Design: MT-only Static Shift Inversion
Core: 200m horizontal x=±10.2km, deep cells ~500km total depth (geometric expansion)
Mesh nodes built explicitly (S1-style) for full control of cell sizes.
"""
import numpy as np
import matplotlib.pyplot as plt
from discretize import TensorMesh

# =============================================================================
# 1. Horizontal Mesh
# =============================================================================
dh_x = 200.0           # core cell size 200m
nx_core = 102           # 102 cells → 20.4km (±10.2km)
n_pad = 12

padding_x = [(dh_x, n_pad, -1.3), (dh_x, nx_core), (dh_x, n_pad, 1.3)]

# =============================================================================
# 2. Depth Mesh — node-based construction for full control
# =============================================================================
# Core region nodes (surface z=0 downward)
# 5m cells for top 80m
z_core = np.concatenate([
    np.arange(0, 80 + 1e-9, 5),           # 0 to -80m:  5m cells (16 cells)
    np.arange(80, 680 + 1e-9, 40)[1:],    # -80 to -680m: 40m cells
    np.arange(680, 3680 + 1e-9, 100)[1:], # -680 to -3680m: 100m cells
    np.arange(3680, 9680 + 1e-9, 200)[1:],# -3680 to -9680m: 200m cells
])

# Transition: -9680m to -12000m, 500m cells
z_trans = np.arange(9680, 12000 + 1e-9, 500)[1:]  # skip first (overlaps with z_core)

# Deep geometric: start at 2000m, x2.0 expansion, 8 cells
# Cell widths: 2, 4, 8, 16, 32, 64, 128, 256 km → total = 510 km
geo_widths = 2000.0 * (2.0 ** np.arange(8))
geo_cumsum = np.cumsum(geo_widths)
z_geo_start = 12000.0  # start from -12km
z_geo = z_geo_start + geo_cumsum

# All z nodes (negative = below surface)
z_nodes = np.concatenate([z_core, z_trans, z_geo])
z_nodes = -z_nodes  # flip to negative z (surface=0 at top)
z_nodes = np.sort(z_nodes)  # sort ascending (most negative → least negative)

hz = np.diff(z_nodes)

# =============================================================================
# 3. Build Mesh
# =============================================================================
mesh = TensorMesh([padding_x, hz], x0=[mesh_x0 := None, z_nodes[0]])
# x0 is set as the origin (bottom-left corner of the mesh)
# For x: let TensorMesh handle it with padding_x
# For z: explicitly set bottom node

# Get x0 for the final mesh
if mesh_x0 is None:
    _tmp = TensorMesh([padding_x, [1.0]], x0="CN")
    x0_x = _tmp.x0[0]
else:
    x0_x = mesh_x0

# Compute x0 correctly
mesh.x0 = np.r_[x0_x, z_nodes[0]]

# =============================================================================
# 4. Mesh Info
# =============================================================================
print("=" * 60)
print("Mesh Design: MT Static Shift Inversion")
print("=" * 60)
print(f"  Total cells:      {mesh.n_cells:,}")
print(f"  Horizontal cells: {mesh.shape_cells[0]} (core {nx_core} + padding {2*n_pad})")
print(f"  Depth cells:      {mesh.shape_cells[1]}")
print(f"  Total depth:      {abs(mesh.nodes_y[-1] - mesh.nodes_y[0])/1000:.1f} km")
print(f"  Surface at z:     {mesh.nodes_y[-1]:.1f} m")
print(f"  Bottom at z:      {mesh.nodes_y[0]:.1f} m")
print(f"  Horizontal range: {mesh.nodes_x.min()/1000:.1f} ~ {mesh.nodes_x.max()/1000:.1f} km")
print(f"  Min dx:           {mesh.h[0].min():.1f} m")
print(f"  Min dz:           {mesh.h[1].min():.1f} m (near-surface)")
print(f"  Max dz:           {mesh.h[1].max()/1000:.1f} km (bottom)")
print(f"  Core depth cells: {len(z_core)}")
print(f"  Deep geo cells:   {len(geo_widths)} (widths: {[f'{w/1000:.0f}km' for w in geo_widths]})")

# Skin depth check
rho_bg = 100
for f in [0.001, 0.01, 0.1]:
    sk = 503 * np.sqrt(rho_bg / f) / 1000  # km
    ratio = abs(z_nodes[-1] - z_nodes[0])/1000 / sk
    print(f"  Skin depth {f:.4f} Hz: {sk:.0f} km, mesh covers {ratio:.1f}x")
print("=" * 60)

# =============================================================================
# 5. Visualization (only when run directly)
# =============================================================================
if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- Left: full mesh overview ---
    mesh.plot_grid(ax=ax1, color='lightgray', lw=0.3)
    ax1.set_xlim(-15000, 15000)
    ax1.set_ylim(-20000, 2000)
    ax1.set_title("Full Mesh Overview (near core)")
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Depth (m)")

    # --- Right: near-surface zoom ---
    mesh.plot_grid(ax=ax2, color='lightgray', lw=0.5)
    ax2.set_xlim(-12000, 12000)
    ax2.set_ylim(-300, 200)
    ax2.set_title("Near-Surface Zoom")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Depth (m)")
    ax2.axhline(y=-80, color='red', linestyle='--', linewidth=1.5,
                label='z=-80m (bottom of 10m layer)')
    ax2.legend()

    plt.tight_layout()
    fig.savefig("mesh_design.png", dpi=150)
    print("\nMesh figure saved: mesh_design.png")
    plt.close(fig)

    # --- Full depth view ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    mesh.plot_grid(ax=ax, color='lightgray', lw=0.3)
    ax.set_xlim(-30000, 30000)
    ax.set_ylim(mesh.nodes_y.min(), 5000)
    ax.set_title("Full Mesh (Deep Extent)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    fig.savefig("mesh_design_full.png", dpi=150)
    print("Full mesh figure saved: mesh_design_full.png")
    plt.close(fig)

    # --- Station Layout ---
    x_mt = np.arange(-10000, 10001, 500.0)
    z_mt = np.zeros_like(x_mt)

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    mesh.plot_grid(ax=ax, color='lightgray', lw=0.4)
    ax.scatter(x_mt, z_mt, marker='s', color='blue', s=50,
               label=f'MT Stations ({len(x_mt)} stn, 500m spacing)', zorder=5)
    ax.set_xlim(-15000, 15000)
    ax.set_ylim(-15000, 1000)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Mesh & MT Survey Station Layout")
    ax.legend(loc='upper right')
    fig.savefig("mesh_survey_layout.png", dpi=150)
    print("Survey layout saved: mesh_survey_layout.png")
    plt.close(fig)
