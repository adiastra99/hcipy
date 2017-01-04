import numpy as np
import copy
from .grid import Grid

class CartesianGrid(Grid):
	coordinate_system = 'cartesian'

	@property
	def x(self):
		return self.coords[0]

	@property
	def y(self):
		return self.coords[1]

	@property
	def z(self):
		return self.coords[2]

	@property
	def w(self):
		return self.coords[3]
	
	def __imul__(self, f):
		self.coords *= f
		self.weights *= np.abs(f)**(self.ndim)
		return self
	
	@staticmethod
	def get_automatic_weights(coords):
		if coords.is_regular:
			return np.prod(coords.delta)
		elif coords.is_separable:
			weights = []
			for i in range(grid.ndim):
				x = coords.separated_coords[i]
				w = (x[2:] - x[:-2]) / 2.
				w = np.concatenate(([x[1] - x[0]], w, [x[-1] - x[-2]]))
				weights.append(w)
			
			return np.multiply.reduce(np.ix_(*weights))