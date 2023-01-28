import numpy as np


class Activation:
    pass


class Layer:
    def __init__(self, shape: tuple[int, int], activation: Activation) -> None:
        self.shape = shape
        self.activation
        self._init_weigths_biases()

    def _init_weigths_biases(self):
        self.w = self._init_array(*self.shape)
        self.b = self._init_array(self.shape[0], 1)

    @staticmethod
    def _init_array(n: int, m: int) -> np.ndarray:
        return np.random.normal(size=(n, m)) * np.sqrt(1 / max(n, m))

    def forward_prop(self, a_: np.ndarray) -> np.ndarray:
        z = self.w.dot(a_) + self.b
        a = self.activation(z)
        return a

    def back_prop(self, dz: np.ndarray) -> None:
        pass


class FinalLayer(Layer):
    def back_prop(self) -> None:
        pass
