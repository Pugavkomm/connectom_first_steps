import sys
import os
sys.path.append("..")
from connectom.node_model import Kuramoto
from connectom.utils import LoadConnectomData
data_path = os.path.join("../", "data")
import networkx as nx
import tensorflow as tf
import numpy as np
#tf.debugging.disable_traceback_filtering()
#tf.config.set_visible_devices([], 'GPU')
loader = LoadConnectomData(data_path)
G = nx.read_graphml(loader(1))
adj_matr = nx.to_numpy_array(G)
kappa = 1.0
# self_freq = 1.0
self_freq = tf.cast(tf.ones(len(adj_matr)), dtype=tf.float32)
# self_freq

import time
Kuramoto.data_type=tf.float32
kuramoto = Kuramoto(
    num_nodes=len(adj_matr), adjacency=adj_matr, kappa=kappa, self_freq=self_freq
)

num_nodes = len(adj_matr)
kuramoto.setup_integrator(step=0.001, num_steps=1000)
# kuramoto.summary()
start = time.perf_counter()
res = kuramoto.run(tf.random.normal((num_nodes,), dtype=tf.float32))
print(time.perf_counter() - start)
print(res.shape)