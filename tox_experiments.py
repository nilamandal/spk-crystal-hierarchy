import numpy as np
import tensorflow as tf
from ogb.graphproppred import Evaluator, GraphPropPredDataset
from tensorflow.keras.layers import BatchNormalization, Dropout, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.datasets.ogb import OGB
from spektral.layers import GCNConv
from spektral.transforms import AdjToSpTensor, GCNFilter

# Load data
dataset_name = "ogbg-moltox21"
dataset = OGB(GraphPropPredDataset(dataset_name))
print(dataset)
graph = dataset[0]
print(graph)
print(graph.y)

F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target

print(F, S, n_out)
