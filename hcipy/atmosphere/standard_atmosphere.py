from .infinite_atmospheric_layer import InfiniteAtmosphericLayer
from .atmospheric_model import MultiLayerAtmosphere

import numpy as np

def make_standard_atmospheric_layers(input_grid, L0=10):
	heights = np.array([500, 1000, 2000, 4000, 8000, 16000])
	velocities = np.array([10, 10, 10, 10, 10, 10])
	Cn_squared = np.array([0.2283, 0.0883, 0.0666, 0.1458, 0.3350, 0.1350]) * 1e-12

	layers = []
	for h, v, cn in zip(heights, velocities, Cn_squared):
		layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0, v, h, 2))

	return layers

def make_atmosphere(input_grid):
    #Uses the Guyon/Males 2017 model. (Unsure of implementation, e.g. outer scales)
    heights = np.array([500, 1000, 2000, 4000, 8000, 16000, 32000])
    velocities = np.array([[0.6541, 6.467], [0.005126, 6.55], [-0.6537, 6.568], [-1.326, 6.568],
                           [-21.98, 0.9], [-9.484, -0.5546], [-5.53, -0.8834]])
    outer_scales = np.array([2, 20, 20, 20, 30, 40, 40])
    Cn_squared = np.array([0.672, 0.051, 0.028, 0.106, 0.08, 0.052, 0.012]) * 1e-12
    layers = []
    for h, v, o, cn in zip(heights, velocities, outer_scales, Cn_squared):
        l = InfiniteAtmosphericLayer(input_grid, cn, o, v, h, 2)
        layers.append(l)

    return layers
