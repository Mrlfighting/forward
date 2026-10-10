"""静态点去常数幅值的对数视电阻率形态及完整协方差白化。"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from simpeg.simulation import BaseSimulation


def helmert_contrasts(n_frequency: int) -> np.ndarray:
    """返回行正交且每行之和为零的 Helmert 对比矩阵。"""

    if n_frequency < 2:
        raise ValueError("形态信息至少需要两个频率")
    matrix = np.zeros((n_frequency - 1, n_frequency))
    for row in range(n_frequency - 1):
        count = row + 1
        scale = np.sqrt(count * (count + 1.0))
        matrix[row, :count] = 1.0 / scale
        matrix[row, count] = -count / scale
    return matrix


class GLSShapeOnlySimulation(BaseSimulation):
    """只输出静态点的白化形态，不引入其绝对视电阻率。"""

    def __init__(
        self,
        base_simulation,
        shape_survey,
        frequencies: np.ndarray,
        n_stations: int,
        static_indices: np.ndarray,
        reference_data: np.ndarray,
        reference_standard_deviation: np.ndarray,
    ):
        self.base_simulation = base_simulation
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.n_stations = int(n_stations)
        self.static_indices = np.asarray(static_indices, dtype=int)
        super().__init__(survey=shape_survey)
        if self.survey.nD != self.n_shape:
            raise ValueError("shape 接收器数量与形态数据数量不一致")
        rho, _ = self._split_full(reference_data)
        rho_std, _ = self._split_full(reference_standard_deviation)
        relative_std = rho_std[:, self.static_indices] / np.abs(rho[:, self.static_indices])
        contrasts = helmert_contrasts(self.frequencies.size)
        operators = []
        for site in range(self.static_indices.size):
            covariance = contrasts @ np.diag(relative_std[:, site] ** 2) @ contrasts.T
            operators.append(np.linalg.solve(np.linalg.cholesky(covariance), contrasts))
        self.shape_operators = np.asarray(operators)
        if np.max(np.abs(self.shape_operators @ np.ones(self.frequencies.size))) > 1e-10:
            raise AssertionError("shape 算子没有消除跨频常数幅值")

    @property
    def model(self):
        return self.base_simulation.model

    @model.setter
    def model(self, value):
        self.base_simulation.model = value

    @property
    def n_shape(self) -> int:
        return (self.frequencies.size - 1) * self.static_indices.size

    def fields(self, m=None):
        return self.base_simulation.fields(m)

    def _split_full(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blocks = np.asarray(vector).reshape(self.frequencies.size, 2, self.n_stations)
        return blocks[:, 0, :], blocks[:, 1, :]

    def _shape(self, full_vector: np.ndarray) -> np.ndarray:
        rho, _ = self._split_full(full_vector)
        selected = rho[:, self.static_indices]
        if np.any(selected <= 0):
            raise ValueError("计算对数形态要求视电阻率全部为正")
        log_rho = np.log(selected)
        return np.concatenate([
            operator @ log_rho[:, site] for site, operator in enumerate(self.shape_operators)
        ])

    def transform_observed(self, full_data: np.ndarray) -> np.ndarray:
        """从完整观测中提取两个静态点的去常数幅值形态。"""

        return self._shape(full_data)

    def transform_standard_deviation(self, shape_weight: float) -> np.ndarray:
        """白化后各形态分量的标准差统一为权重倒数。"""

        if shape_weight <= 0:
            raise ValueError("shape 权重必须为正")
        return np.full(self.n_shape, 1.0 / shape_weight)

    def dpred(self, m=None, f=None):
        return self._shape(np.asarray(self.base_simulation.dpred(m, f=f), dtype=float))

    def _transform_jvec(self, full_jvec: np.ndarray, full_prediction: np.ndarray) -> np.ndarray:
        rho, _ = self._split_full(full_prediction)
        jrho, _ = self._split_full(full_jvec)
        log_jvec = jrho[:, self.static_indices] / rho[:, self.static_indices]
        return np.concatenate([
            operator @ log_jvec[:, site] for site, operator in enumerate(self.shape_operators)
        ])

    def Jvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m)
        prediction = self.base_simulation.dpred(m, f=f)
        return self._transform_jvec(self.base_simulation.Jvec(m, v, f=f), prediction)

    def Jvec_approx(self, m, v, f=None):
        if f is None:
            f = self.fields(m)
        prediction = self.base_simulation.dpred(m, f=f)
        return self._transform_jvec(self.base_simulation.Jvec_approx(m, v, f=f), prediction)

    def _adjoint_to_full(self, transformed: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        shape = np.asarray(transformed).reshape(self.static_indices.size, self.frequencies.size - 1)
        full = np.zeros((self.frequencies.size, 2, self.n_stations))
        rho, _ = self._split_full(prediction)
        for site, (index, operator) in enumerate(zip(self.static_indices, self.shape_operators)):
            full[:, 0, index] = (operator.T @ shape[site]) / rho[:, index]
        return full.ravel()

    def Jtvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m)
        prediction = self.base_simulation.dpred(m, f=f)
        return self.base_simulation.Jtvec(m, self._adjoint_to_full(v, prediction), f=f)

    def Jtvec_approx(self, m, v, f=None):
        if f is None:
            f = self.fields(m)
        prediction = self.base_simulation.dpred(m, f=f)
        return self.base_simulation.Jtvec_approx(m, self._adjoint_to_full(v, prediction), f=f)

    def getJtJdiag(self, m, W=None, f=None):
        """向完整模拟器传播 shape 残差的对角灵敏度权重。"""

        if f is None:
            f = self.fields(m)
        prediction = self.base_simulation.dpred(m, f=f)
        weights = np.ones(self.n_shape) if W is None else np.asarray(W.diagonal())
        shape_w = weights.reshape(self.static_indices.size, self.frequencies.size - 1)
        full_w2 = np.zeros((self.frequencies.size, 2, self.n_stations))
        rho, _ = self._split_full(prediction)
        for site, (index, operator) in enumerate(zip(self.static_indices, self.shape_operators)):
            coefficients = shape_w[site, :, None] * operator
            full_w2[:, 0, index] = np.sum(coefficients**2, axis=0) / rho[:, index] ** 2
        return self.base_simulation.getJtJdiag(m, W=sp.diags(np.sqrt(full_w2.ravel())), f=f)
