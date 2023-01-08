from turtle import update
import mnist
from matplotlib import pyplot
import numpy as np

x_train, x_test = mnist.train_images(), mnist.test_images()
y_train, y_test = mnist.train_labels(), mnist.test_labels()


def plot_random_digs(digs: np.array, dig_labels: np.array, n: int = 10) -> None:
    for _ in range(10):
        idx = np.random.randint(0, digs.shape[0])
        plot_dig(digs[idx], dig_labels[idx])


def plot_dig(dig_array: np.ndarray, label: int) -> None:
    pyplot.imshow(dig_array, cmap=pyplot.get_cmap("gray"))
    pyplot.title(f"label = {label}")
    pyplot.show()


# plot_random_digs(x_test, y_test)


def transform_reshape_dig_images(digs: np.ndarray) -> np.ndarray:
    digs = reshape_dig_images(digs)
    digs = transform_dig_images(digs)
    return digs


def transform_dig_images(digs: np.ndarray) -> np.ndarray:
    return digs.astype(float) / np.max(digs)


def reshape_dig_images(digs: np.ndarray) -> np.ndarray:
    return digs.reshape(digs.shape[0], digs.shape[1] * digs.shape[2]).T


def one_hot_labels(labels: np.ndarray) -> np.ndarray:
    n_labels = labels.size
    one_hot_encoded = np.zeros((n_labels, np.unique(labels).size))
    one_hot_encoded[np.arange(n_labels), labels] = 1
    return one_hot_encoded.T


def gradient_descent(x, y, iterations, alpha):

    w1, b1 = init_layer_params(10, x.shape[0])
    w2, b2 = init_layer_params(10, 10)

    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(x, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = back_prop(x, z1, a1, z2, a2, w2, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        if (i % 10) == 0:
            print(i)
            print(f"Accuracy : {get_accuracy(y, a2)}")

    return w1, b1, w2, b2


def forward_prop(a0, w1, b1, w2, b2):

    z1 = w1.dot(a0) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2


def back_prop(
    a0: np.array, z1: np.ndarray, a1: np.ndarray, z2: np.ndarray, a2: np.ndarray, w2: np.ndarray, labels: np.ndarray
):

    n_examples = labels.size

    dz2 = a2 - labels
    dw2 = dz2.dot(a1.T) / n_examples
    db2 = np.sum(dz2) / n_examples

    dz1 = w2.T.dot(dz2) * relu_deriv(z1)
    dw1 = dz1.dot(a0.T) / n_examples
    db1 = np.sum(dz1) / n_examples

    return dw1, db1, dw2, db2


def init_layer_params(n: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    w = np.random.normal(size=(n, m)) * np.sqrt(1 / m)
    b = np.random.normal(size=(n, 1)) * np.sqrt(1 / n)
    return w, b


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    return w1, b1, w2, b2


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def relu_deriv(x: np.array) -> int:
    return (x > 0).astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    x_exp = np.exp(x - np.max(x))
    return x_exp / x_exp.sum(axis=0)


def get_accuracy(y, y_hat):
    return np.sum(np.argmax(y_hat, axis=0) == np.argmax(y, axis=0)) / y.shape[1]


x_train, x_test = transform_reshape_dig_images(x_train), transform_reshape_dig_images(x_test)
y_train, y_test = one_hot_labels(y_train), one_hot_labels(y_test)
w1, b1, w2, b2 = gradient_descent(x_train, y_train, 5000, 0.1)
