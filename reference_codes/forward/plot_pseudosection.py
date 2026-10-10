"""
Pseudosection Plotting (TE Mode): MT Static Shift Inversion
TE mode: Z_xy = E_x / H_y  (Simulation2DElectricField)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
from simpeg.utils import BreakingChangeWarning
from simpeg.electromagnetics import natural_source as nsem
from build_mesh import mesh
from build_model import m_true, mapping

warnings.filterwarnings(action='ignore', category=BreakingChangeWarning)

# =============================================================================
# 1. Build MT Survey — 10 km range, 500m spacing → 41 stations
# =============================================================================
def build_mt_survey(freqs, rx_x_range, rx_spacing, mode="TE"):
    rx_x = np.arange(rx_x_range[0], rx_x_range[1] + 1, rx_spacing)
    rx_locs = np.c_[rx_x, np.zeros_like(rx_x)]
    src_list = []
    orientation = "xy" if mode == "TE" else "yx"
    for f in freqs:
        rx_real = nsem.receivers.Impedance(
            rx_locs, orientation=orientation, component="real"
        )
        rx_imag = nsem.receivers.Impedance(
            rx_locs, orientation=orientation, component="imag"
        )
        src_list.append(nsem.sources.Planewave([rx_real, rx_imag], f))
    return nsem.survey.Survey(src_list)


freqs = np.logspace(-3, 2.5, 20)
n_freq = len(freqs)
station_xs = np.arange(-10000, 10001, 500.0)
n_stn = len(station_xs)

survey = build_mt_survey(freqs, [-10000, 10000], 500.0, mode="TE")

print("=" * 60)
print("TE Mode Pseudosection — 10 km Range")
print("=" * 60)
print(f"  Frequencies:  {n_freq} ({freqs[0]:.4f} ~ {freqs[-1]:.0f} Hz)")
print(f"  Stations:     {n_stn} ({station_xs[0]:.0f} ~ {station_xs[-1]:.0f} m)")
print(f"  Total data:   {n_freq * n_stn * 2}")

# =============================================================================
# 2. Forward Model
# =============================================================================
print("\n[1/2] Running TE forward model...")
sim = nsem.simulation.Simulation2DElectricField(
    mesh, survey=survey, sigmaMap=mapping, storeInnerProduct=True
)
dpred = sim.dpred(m_true)

# =============================================================================
# 3. Compute apparent resistivity and phase
# =============================================================================
print("[2/2] Computing rho_a and phase...")

mu_0 = 4.0 * np.pi * 1e-7
rho_a = np.full((n_freq, n_stn), np.nan)
phase = np.full((n_freq, n_stn), np.nan)

for i_f in range(n_freq):
    omega = 2.0 * np.pi * freqs[i_f]
    for i_s in range(n_stn):
        idx_real = i_f * 2 * n_stn + i_s
        idx_imag = i_f * 2 * n_stn + n_stn + i_s
        zc = complex(-dpred[idx_real], -dpred[idx_imag])  # sign convention
        rho_a[i_f, i_s] = abs(zc) ** 2 / (mu_0 * omega)
        phase[i_f, i_s] = np.arctan2(zc.imag, zc.real) * 180.0 / np.pi

# Flip so row 0 = highest frequency (pseudosection: high-f at top)
rho_a = rho_a[::-1, :]
phase = phase[::-1, :]
freqs_plot = freqs[::-1]  # high → low

# =============================================================================
# 4. Y-axis edges in log10(frequency) space
# =============================================================================
logf = np.log10(freqs_plot)
logf_edges = np.empty(n_freq + 1)
logf_edges[1:-1] = 0.5 * (logf[:-1] + logf[1:])
dh = np.mean(np.abs(np.diff(logf))) / 2
logf_edges[0] = logf[0] + dh
logf_edges[-1] = logf[-1] - dh

# X-axis edges
dx = station_xs[1] - station_xs[0]
x_edges = np.append(station_xs - dx / 2, station_xs[-1] + dx / 2)

# =============================================================================
# 5. Plot — colormap style from ck_plot_pseudosection.py
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

cmap_rho = "jet_r"
cmap_phs = "jet_r"

# Data-driven color ranges
rho_vmin = np.nanmin(rho_a) * 0.8
rho_vmax = np.nanmax(rho_a) * 1.2
norm_rho = mcolors.LogNorm(vmin=rho_vmin, vmax=rho_vmax)

pha_vmin = np.floor(np.nanmin(phase))
pha_vmax = np.ceil(np.nanmax(phase))
norm_pha = mcolors.Normalize(vmin=pha_vmin, vmax=pha_vmax)

# X-axis in km
x_plot = station_xs / 1000.0

# --- (a) Apparent Resistivity Pseudosection ---
ax = axes[0, 0]
levels_rho = np.logspace(np.log10(rho_vmin), np.log10(rho_vmax), 50)
im1 = ax.contourf(x_plot, logf, rho_a,
                   levels=levels_rho, cmap=cmap_rho, norm=norm_rho, extend='both')
ytick_freqs = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
ax.set_yticks(np.log10(ytick_freqs))
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('(a) Apparent Resistivity — TE Mode ($Z_{xy}$)')
cb1 = plt.colorbar(im1, ax=ax, label='App. Resistivity (Ohm-m)')

# --- (b) Phase Pseudosection (contourf for smooth physics) ---
ax = axes[0, 1]
levels_pha = np.linspace(pha_vmin, pha_vmax, 50)
im2 = ax.contourf(x_plot, logf, phase,
                   levels=levels_pha, cmap=cmap_phs, norm=norm_pha, extend='both')
ax.set_yticks(np.log10(ytick_freqs))
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('(b) Phase — TE Mode ($Z_{xy}$)')
cb2 = plt.colorbar(im2, ax=ax, label='Phase (deg)')

# --- (c) Sounding Curves ---
ax = axes[1, 0]
plot_stn = [-5000, -3000, 0, 3000, 5000]
colors = ['darkred', 'red', 'black', 'blue', 'darkblue']
for stn_x, color in zip(plot_stn, colors):
    i_s = np.argmin(np.abs(station_xs - stn_x))
    ax.loglog(freqs, rho_a[::-1, i_s], 'o-', color=color, markersize=4,
              linewidth=1.5, label=f'x = {stn_x:+d} m')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('App. Resistivity (Ohm-m)')
ax.set_title('(c) Sounding Curves — TE Mode')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

# --- (d) Phase Sounding Curves ---
ax = axes[1, 1]
for stn_x, color in zip(plot_stn, colors):
    i_s = np.argmin(np.abs(station_xs - stn_x))
    ax.semilogx(freqs, phase[::-1, i_s], 's-', color=color, markersize=4,
                linewidth=1.5, label=f'x = {stn_x:+d} m')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Phase (deg)')
ax.set_title('(d) Phase Sounding — TE Mode')
ax.set_ylim(0, 90)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

plt.tight_layout()
fig.savefig('pseudosection_te.png', dpi=150)
print('\nSaved: pseudosection_te.png')
plt.close(fig)

print()
print("=" * 60)
print("TE Data Summary")
print("=" * 60)
print(f"  Rho_a:  {np.nanmin(rho_a):.1f} ~ {np.nanmax(rho_a):.1f} Ohm-m")
print(f"  Phase:  {np.nanmin(phase):.1f} ~ {np.nanmax(phase):.1f} deg")
print("=" * 60)

# =============================================================================
# 6. Save forward data for later inversion use
# =============================================================================
import os
os.makedirs('forward_data', exist_ok=True)
np.save('forward_data/dpred_41stn.npy', np.asarray(dpred, dtype=np.float64).ravel())
np.save('forward_data/stations_41stn.npy', station_xs)
np.save('forward_data/freqs.npy', freqs)
print(f"\nForward data saved: forward_data/ ({len(station_xs)} stn x {len(freqs)} freq x 2 = {2*len(freqs)*len(station_xs)} points)")

