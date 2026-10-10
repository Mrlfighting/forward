"""论文 2.1 正演的固定物理与观测参数。"""

import numpy as np


BACKGROUND_RESISTIVITY_OHM_M = 100.0
FREQUENCIES_HZ = np.logspace(-3, 2.5, 20)
STATION_X_M = np.arange(-10000.0, 10000.0 + 1.0, 500.0)
VACUUM_PERMEABILITY = 4.0 * np.pi * 1e-7
