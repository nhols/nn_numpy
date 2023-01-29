from typing import Callable
import numpy as np

from t import forward_prop


class Activation:
    def __call__(self, x: np.ndarray) -> None:
        raise NotImplementedError(f"Activation function {type(self)} not implemented")

    def deriv(self, x: np.ndarray) -> None:
        raise NotImplementedError(f"deriv method not implementd for activation function: {type(self)}")


class ReLU(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def deriv(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Softmax(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_exp = np.exp(x - np.max(x))
        return x_exp / x_exp.sum(axis=0)


LayerDesc = tuple[int, Activation]


class NumpyNN:
    def __init__(self, dim_in: int, architecture: list[LayerDesc], dim_out: int) -> None:

        self.layers: list[Layer | FinalLayer] = []

        prev_n = dim_in
        for n_layer, activation in architecture[:-1]:
            layer = Layer(shape=(prev_n, n_layer), activation=activation)
            self.layers.add(layer)
            prev_n = n_layer

        dim_out, final_activation = architecture[-1]
        final_layer = FinalLayer(shape=(prev_n, dim_out), activation=final_activation)
        self.layers.add(final_layer)

    def forward_prop(self, input: np.ndarray):
        for layer in self.layers:
            layer.forward_prop(input)
            input = layer.a

    def back_prop(self, labels: np.ndarray):
        for layer in layers[::-1]:
            pass


class Layer:
    def __init__(self, shape: tuple[int, int], activation: Activation) -> None:
        self.shape = shape
        self.activation = activation
        self._init_weigths_biases()
        self.z = None
        self.dz = None
        self.a = None

    def _init_weigths_biases(self):
        self.w = self._init_array(*self.shape)
        self.b = self._init_array(self.shape[0], 1)

    @staticmethod
    def _init_array(n: int, m: int) -> np.ndarray:
        return np.random.normal(size=(n, m)) * np.sqrt(1 / max(n, m))

    def forward_prop(self, a_prev: np.ndarray) -> None:
        self.z = self.w.dot(a_prev) + self.b
        self.a = self.activation(self.z)

    def back_prop(
        self, w_next: np.ndarray, dz_next: np.ndarray, a_prev: np.ndarray, n_samples: int, alpha: float
    ) -> None:
        self._calc_dz(w_next, dz_next)
        self._grad_descent(a_prev, n_samples, alpha)

    def _grad_descent(self, a_prev: np.ndarray, n_samples: int, alpha: float) -> None:
        dw = self._calc_dw(a_prev, n_samples)
        db = self._calc_db(n_samples)
        self._update_weights(dw, db, alpha)

    def _calc_dz(self, w_next: np.ndarray, dz_next: np.ndarray) -> None:
        self.dz = w_next.T.dot(dz_next) * self.activation.deriv(self.z)

    def _calc_dw(self, a_prev: np.ndarray, n_samples: int) -> np.ndarray:
        return self.dz.dot(a_prev.T) / n_samples

    def _calc_db(self, n_samples: int) -> np.ndarray:
        return np.sum(self.dz) / n_samples

    def _update_weights_biases(self, dw, db, alpha) -> None:
        self.w -= alpha * dw
        self.b -= alpha * db


class FinalLayer(Layer):
    def _calc_dz(self, labels: np.ndarray) -> None:
        self.dz = self.a - labels
