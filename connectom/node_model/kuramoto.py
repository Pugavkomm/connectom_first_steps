import tensorflow as tf
import math
import numpy as np


@tf.function(reduce_retracing=True)
def _kuramoto_rigth_part(
    frequencies, kappa, adjacency, data, num_nodes, multiplier
) -> tf.Variable:

    A = tf.expand_dims(data, -1) * multiplier
    delta_phases = A - tf.transpose(A)

    return (
        frequencies
        - kappa
        * tf.reduce_sum(tf.matmul(adjacency, tf.math.sin(delta_phases)), axis=1)
        / num_nodes
    )


class Kuramoto:
    data_type = tf.float32

    def __init__(
        self,
        num_nodes: int,
        adjacency: tf.Tensor | np.ndarray,
        kappa: tf.Tensor,
        self_freq: np.ndarray | tf.Tensor | float | int,
    ) -> None:
        self.num_nodes = num_nodes
        self.adjacency = tf.constant(adjacency, dtype=self.data_type)
        self.kappa = tf.constant(kappa, dtype=self.data_type)
        self.setup_integrator()  # by default
        if isinstance(self_freq, tf.Tensor) or isinstance(self_freq, np.ndarray):
            if (len(self_freq) != 1) and (len(self_freq) != num_nodes):
                raise ValueError(
                    f"""Self frequency must be scalar or vector, 
                    which len(vector) == num_nodes.
                    Expected: 1 | {num_nodes}, but actually: {len(self_freq)}"""
                )
        self.self_freq = tf.constant(self_freq, dtype=self.data_type)

    def setup_integrator(self, step: float | tf.Tensor = 0.001, num_steps: int = 10000):
        self.step = tf.constant(step, dtype=self.data_type)
        self.num_steps = int(num_steps)

    def summary(self):
        print("Kuramoto model has: ")
        print("-" * 50)
        print("Nodes parameters:")
        print(f"Number of node: {self.num_nodes}")
        print(
            f"Number of edjes: {tf.reduce_sum(tf.cast(self.adjacency, dtype=tf.int64)).numpy() // 2}"
        )
        # if isinstance(self.self_freq, tf.Tensor) or isinstance(
        #    self.self_freq, np.ndarray
        # ):
        print(f"Self frequencies: {self.self_freq}")
        print("-" * 50)
        print("Integration parameters:")
        print(f"Step: {self.step}")
        print(f"Number of steps: {self.num_steps}")

    # @tf.function(reduce_retracing=True)
    def _rk4(self, result: tf.Tensor):

        one_per_six = 1 / 6
        multiplier = tf.ones((self.num_nodes, self.num_nodes), dtype=self.data_type)
        for i in range(self.num_steps):

            k1 = (
                _kuramoto_rigth_part(
                    self.self_freq,
                    self.kappa,
                    self.adjacency,
                    result[:, -1],
                    self.num_nodes,
                    multiplier,
                )
                * self.step
            )
            k2 = (
                _kuramoto_rigth_part(
                    self.self_freq,
                    self.kappa,
                    self.adjacency,
                    result[:, -1] + k1 / 2,
                    self.num_nodes,
                    multiplier,
                )
                * self.step
            )
            k3 = (
                _kuramoto_rigth_part(
                    self.self_freq,
                    self.kappa,
                    self.adjacency,
                    result[:, -1] + k2 / 2,
                    self.num_nodes,
                    multiplier,
                )
                * self.step
            )
            k4 = (
                _kuramoto_rigth_part(
                    self.self_freq,
                    self.kappa,
                    self.adjacency,
                    result[:, -1] + k3,
                    self.num_nodes,
                    multiplier,
                )
                * self.step
            )
            tmp_res = tf.expand_dims(
                result[:, -1] + one_per_six * (k1 + 2 * k2 + 2 * k3 + k4), -1
            )
            tmp_res = (
                tmp_res
                + (
                    -tf.cast(tmp_res > math.pi, dtype=self.data_type)
                    + tf.cast(tmp_res < -math.pi, dtype=self.data_type)
                )
                * 2
                * math.pi
            )

            result = tf.concat((result, tmp_res), axis=1)
        return result

    def run(self, initial_conditions: tf.Tensor | np.ndarray):
        result = initial_conditions = tf.reshape(
            tf.constant(initial_conditions, dtype=self.data_type), (self.num_nodes, 1)
        )
        print("Run calculation...")
        print(
            f"""You need to have  of memory without overhead {
                (self.num_nodes * self.num_steps * 1e-6 * (
                    4 if self.data_type == tf.float32 else 8))}MB"""
        )
        result = tf.Variable(tf.zeros(shape=(self.num_nodes, self.num_steps + 1)))
        result[:, 0].assign(initial_conditions)

        return self._rk4(result)
