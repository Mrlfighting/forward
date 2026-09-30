"""构建论文 2.1 的二维张量网格。"""

import numpy as np
from discretize import TensorMesh


def build_mesh() -> TensorMesh:
    """以 100 m 水平单元及 200 m 深度内的 5 m 单元表示浅部体。"""
    horizontal_widths = [(200.0, 12, -1.3), (100.0, 204), (200.0, 12, 1.3)]
    shallow_nodes = np.concatenate(
        [
            np.arange(0.0, 200.0 + 1e-9, 5.0),
            np.arange(200.0, 680.0 + 1e-9, 40.0)[1:],
            np.arange(680.0, 3680.0 + 1e-9, 100.0)[1:],
            np.arange(3680.0, 9680.0 + 1e-9, 200.0)[1:],
        ]
    )
    # 原有过渡节点止于 11680 m，下一深部节点为 14000 m。
    transition_nodes = np.arange(9680.0, 12000.0 + 1e-9, 500.0)[1:]
    deep_nodes = 12000.0 + np.cumsum(2000.0 * 2.0 ** np.arange(8))
    vertical_nodes = np.sort(-np.concatenate([shallow_nodes, transition_nodes, deep_nodes]))
    centered_mesh = TensorMesh([horizontal_widths, [1.0]], x0="CN")
    return TensorMesh(
        [horizontal_widths, np.diff(vertical_nodes)],
        x0=[centered_mesh.x0[0], vertical_nodes[0]],
    )
