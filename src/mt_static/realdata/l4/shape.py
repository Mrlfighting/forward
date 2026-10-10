"""沿用旧 L4 的 GLS 幅值不变视电阻率形态变换。"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from simpeg.simulation import BaseSimulation


def helmert_contrasts(n: int) -> np.ndarray:
    """返回与常数向量正交的 Helmert 对比矩阵。"""
    if n < 2:
        raise ValueError("At least two frequencies are required")
    matrix = np.zeros((n - 1, n))
    for row in range(n - 1):
        count = row + 1
        scale = np.sqrt(count * (count + 1.0))
        matrix[row, :count] = 1.0 / scale
        matrix[row, count] = -count / scale
    return matrix


class GLSAmplitudeInvariantShapeSimulation(BaseSimulation):
    """保留普通数据，并加入静态点白化后的对数视电阻率形态。"""

    def __init__(
        self,
        base_simulation,
        transformed_survey,
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
        self.normal_indices = np.setdiff1d(
            np.arange(self.n_stations), self.static_indices
        )
        super().__init__(survey=transformed_survey)
        expected = self.n_regular + self.n_shape
        if transformed_survey.nD != expected:
            raise ValueError(
                f"Transformed survey has {transformed_survey.nD} data; expected {expected}"
            )

        rho, _ = self._split_full(reference_data)
        rho_std, _ = self._split_full(reference_standard_deviation)
        relative_std = rho_std[:, self.static_indices] / np.abs(
            rho[:, self.static_indices]
        )
        contrasts = helmert_contrasts(self.frequencies.size)
        operators = []
        for site in range(self.static_indices.size):
            covariance = contrasts @ np.diag(relative_std[:, site] ** 2) @ contrasts.T
            operators.append(np.linalg.solve(np.linalg.cholesky(covariance), contrasts))
        self.shape_operators = np.asarray(operators)
        invariance_error = float(
            np.max(np.abs(self.shape_operators @ np.ones(self.frequencies.size)))
        )
        if invariance_error > 1.0e-10:
            raise RuntimeError(f"Shape operator invariance error: {invariance_error}")

    @property
    def model(self):
        return self.base_simulation.model

    @model.setter
    def model(self, value):
        self.base_simulation.model = value

    @property
    def n_regular(self) -> int:
        return self.frequencies.size * (
            self.normal_indices.size + self.n_stations
        )

    @property
    def n_shape(self) -> int:
        return (self.frequencies.size - 1) * self.static_indices.size

    def fields(self, m=None):
        return self.base_simulation.fields(m)

    def _split_full(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shaped = np.asarray(vector).reshape(
            self.frequencies.size, 2, self.n_stations
        )
        return shaped[:, 0, :], shaped[:, 1, :]

    def _regular(self, vector: np.ndarray) -> np.ndarray:
        rho, phase = self._split_full(vector)
        return np.concatenate((rho[:, self.normal_indices], phase), axis=1).ravel()

    def _shape(self, full_prediction: np.ndarray) -> np.ndarray:
        rho, _ = self._split_full(full_prediction)
        selected = rho[:, self.static_indices]
        if np.any(selected <= 0.0):
            raise ValueError("Apparent resistivity must be positive")
        log_rho = np.log(selected)
        return np.concatenate(
            [
                operator @ log_rho[:, site]
                for site, operator in enumerate(self.shape_operators)
            ]
        )

    def transform_observed(self, full_data: np.ndarray) -> np.ndarray:
        return np.r_[self._regular(full_data), self._shape(full_data)]

    def transform_standard_deviation(
        self,
        full_data: np.ndarray,
        full_standard_deviation: np.ndarray,
        shape_weight: float = 1.0,
    ) -> np.ndarray:
        if shape_weight <= 0.0:
            raise ValueError("shape_weight must be positive")
        regular_std = self._regular(full_standard_deviation)
        return np.r_[regular_std, np.full(self.n_shape, 1.0 / shape_weight)]

    def dpred(self, m=None, f=None):
        full = np.asarray(self.base_simulation.dpred(m, f=f), dtype=float)
        return np.r_[self._regular(full), self._shape(full)]

    def _transform_jvec(
        self, full_jvec: np.ndarray, full_prediction: np.ndarray
    ) -> np.ndarray:
        rho, _ = self._split_full(full_prediction)
        jrho, _ = self._split_full(full_jvec)
        log_jvec = jrho[:, self.static_indices] / rho[:, self.static_indices]
        shape_jvec = np.concatenate(
            [
                operator @ log_jvec[:, site]
                for site, operator in enumerate(self.shape_operators)
            ]
        )
        return np.r_[self._regular(full_jvec), shape_jvec]

    def Jvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m)
        prediction = self.base_simulation.dpred(m, f=f)
        full_jvec = self.base_simulation.Jvec(m, v, f=f)
        return self._transform_jvec(full_jvec, prediction)

    def Jvec_approx(self, m, v, f=None):
        if f is None:
            f = self.fields(m)
        prediction = self.base_simulation.dpred(m, f=f)
        full_jvec = self.base_simulation.Jvec_approx(m, v, f=f)
        return self._transform_jvec(full_jvec, prediction)

    def _adjoint_to_full(
        self, transformed_vector: np.ndarray, full_prediction: np.ndarray
    ) -> np.ndarray:
        vector = np.asarray(transformed_vector)
        regular = vector[: self.n_regular].reshape(
            self.frequencies.size,
            self.normal_indices.size + self.n_stations,
        )
        shape = vector[self.n_regular :].reshape(
            self.static_indices.size, self.frequencies.size - 1
        )
        full = np.zeros((self.frequencies.size, 2, self.n_stations))
        full[:, 0, self.normal_indices] = regular[:, : self.normal_indices.size]
        full[:, 1, :] = regular[:, self.normal_indices.size :]
        rho, _ = self._split_full(full_prediction)
        for site, (index, operator) in enumerate(
            zip(self.static_indices, self.shape_operators)
        ):
            full[:, 0, index] += (operator.T @ shape[site]) / rho[:, index]
        return full.ravel()

    def Jtvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m)
        prediction = self.base_simulation.dpred(m, f=f)
        return self.base_simulation.Jtvec(
            m, self._adjoint_to_full(v, prediction), f=f
        )

    def Jtvec_approx(self, m, v, f=None):
        if f is None:
            f = self.fields(m)
        prediction = self.base_simulation.dpred(m, f=f)
        return self.base_simulation.Jtvec_approx(
            m, self._adjoint_to_full(v, prediction), f=f
        )

    def getJtJdiag(self, m, W=None, f=None):
        if f is None:
            f = self.fields(m)
        prediction = self.base_simulation.dpred(m, f=f)
        weights = np.ones(self.survey.nD) if W is None else np.asarray(W.diagonal())
        regular_w = weights[: self.n_regular].reshape(
            self.frequencies.size,
            self.normal_indices.size + self.n_stations,
        )
        shape_w = weights[self.n_regular :].reshape(
            self.static_indices.size, self.frequencies.size - 1
        )
        full_w2 = np.zeros((self.frequencies.size, 2, self.n_stations))
        full_w2[:, 0, self.normal_indices] = regular_w[
            :, : self.normal_indices.size
        ] ** 2
        full_w2[:, 1, :] = regular_w[:, self.normal_indices.size :] ** 2
        rho, _ = self._split_full(prediction)
        for site, (index, operator) in enumerate(
            zip(self.static_indices, self.shape_operators)
        ):
            coefficients = shape_w[site, :, None] * operator
            full_w2[:, 0, index] += (
                np.sum(coefficients**2, axis=0) / rho[:, index] ** 2
            )
        return self.base_simulation.getJtJdiag(
            m, W=sp.diags(np.sqrt(full_w2.ravel())), f=f
        )
