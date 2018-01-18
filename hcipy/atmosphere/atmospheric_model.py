from ..optics import OpticalElement
from ..field import Field
from ..propagation import FresnelPropagator

import numpy as np
from scipy.special import gamma, kv

class AtmosphericLayer(OpticalElement):
	def __init__(self, input_grid, Cn_squared=None, L0=np.inf, velocity=0, height=0):

		self.input_grid = input_grid
		self.Cn_squared = Cn_squared
		self.L0 = L0

		self._velocity = None
		self.velocity = velocity

		self.height = height
		self._t = 0
	
	def evolve_until(self, t):
		raise NotImplementedError()
	
	@property
	def t(self):
		return self._t
	
	@t.setter
	def t(self, t):
		self.evolve_until(t)
	
	@property
	def Cn_squared(self):
		return self._Cn_squared
	
	@Cn_squared.setter
	def Cn_squared(self, Cn_squared):
		raise NotImplementedError()

	@property
	def outer_scale(self):
		return self._outer_scale
	
	@outer_scale.setter
	def outer_scale(self, L0):
		raise NotImplementedError()
	
	@property
	def L0(self):
		return self.outer_scale
	
	@L0.setter
	def L0(self, L0):
		self.outer_scale = L0

	@property
	def velocity(self):
		return self._velocity

	@velocity.setter
	def velocity(self, velocity):
		if np.isscalar(velocity):
			if self._velocity is None:
				theta = np.random.rand() * 2 * np.pi
				self._velocity = velocity * np.array([np.cos(theta), np.sin(theta)])
			else:
				self._velocity *= velocity / np.sqrt(np.dot(velocity, velocity))
		else:
			self._velocity = velocity

	def phase_for(self, wavelength):
		raise NotImplementedError()
	
	@property
	def output_grid(self):
		return self.input_grid
	
	def forward(self, wf):
		wf = wf.copy()
		wf.electric_field *= np.exp(1j * self.phase_for(wf.wavelength))
		return wf
	
	def backward(self, wf):
		wf = wf.copy()
		wf.electric_field *= np.exp(-1j * self.phase_for(wf.wavelength))
		return wf

class MultiLayerAtmosphere(OpticalElement):
	def __init__(self, layers, scintilation=False):
		self.layers = layers
		self._scintilation = scintilation
		self._t = 0
		self._dirty = True

		self.calculate_propagators()

	def calculate_propagators(self):
		heights = np.array([l.height for l in self.layers])
		layer_indices = np.argsort(-heights)

		sorted_heights = heights[layer_indices]
		delta_heights = sorted_heights[:-1] - sorted_heights[1:]
		grid = self.layers[0].input_grid

		if self.scintilation:
			propagators = [FresnelPropagator(grid, h) for h in delta_heights]

		self.elements = []
		for i, j in enumerate(layer_indices):
			self.elements.append(self.layers[j])
			if self.scintilation and i < len(propagators):
				self.elements.append(propagators[i])
		
		self._dirty = False
	
	@property
	def layers(self):
		return self._layers

	@layers.setter
	def layers(self, layers):
		self._layers = layers
		self._dirty = True
	
	@property
	def scintilation(self):
		return self._scintilation

	@scintilation.setter
	def scintilation(self, scintilation):
		self._dirty = scintilation != self.scintilation
		self._scintilation = scintilation
	
	def evolve_until(self, t):
		for l in self.layers:
			l.evolve_until(t)
		self._t = t
	
	@property
	def Cn_squared(self):
		return np.sum([l.Cn_squared for l in self.layers])
	
	@Cn_squared.setter
	def Cn_squared(self, Cn_squared):
		old_Cn_squared = self.Cn_squared
		for l in self.layers:
			l.Cn_squared = l.Cn_squared / old_Cn_squared * Cn_squared
	
	@property
	def outer_scale(self):
		return self.layers[0].outer_scale
	
	@outer_scale.setter
	def outer_scale(self, L0):
		for l in self.layers:
			l.outer_scale = L0
	
	@property
	def t(self):
		return self._t
	
	@t.setter
	def t(self, t):
		self.evolve_until(t)

	def forward(self, wavefront):
		if self._dirty:
			self.calculate_propagators()
		
		wf = wavefront.copy()
		for el in self.elements:
			wf = el.forward(wf)
		return wf
	
	def backward(self, wavefront):
		if self._dirty:
			self.calculate_propagators()
		
		wf = wavefront.copy()
		for el in reversed(self.elements):
			wf = el.backward(wf)
		return wf

def von_karman_psd(grid, r0, L0=0.1):
	u = grid.as_('polar').r + 1e-20
	res = 0.0299 * ((u**2 + u_o**2) / (2 * np.pi)**2)**(-11 / 6.)

	res[u < 1e-19] = 0
	return Field(res, grid)

def phase_covariance_von_karman(r0, L0):
	def func(grid):
		r = grid.as_('polar').r + 1e-10
		
		a = (L0 / r0)**(5 / 3)
		b = gamma(11 / 6) / (2**(5 / 6) * np.pi**(8 / 3))
		c = (24 / 5 * gamma(6 / 5))**(5 / 6)
		d = (2 * np.pi * r / L0)**(5 / 6)
		e = kv(5 / 6, 2 * np.pi * r / L0)

		return Field(a * b * c * d * e, grid)
	return func

def phase_structure_function_von_karman(r0, L0):
	def func(grid):
		r = grid.as_('polar').r + 1e-10
		
		a = (L0 / r0)**(5 / 3)
		b = 2**(1 / 6) * gamma(11 / 6) / np.pi**(8 / 3)
		c = (24 / 5 * gamma(6 / 5))**(5 / 6)
		d = gamma(5 / 6) / 2**(1 / 6)
		e = (2 * np.pi * r / L0)**(5 / 6)
		f = kv(5 / 6, 2 * np.pi * r / L0)

		return Field(a * b * c * (d - e * f), grid)
	return func

def power_spectral_density_von_karman(r0, L0):
	def func(grid):
		u = grid.as_('polar').r + 1e-10
		u0 = 2 * np.pi / L0

		res = 0.0229 * ((u**2 + u0**2) / (2 * np.pi)**2)**(-11 / 6.) * r0**(-5 / 3)
		res[u < 1e-9] = 0

		return Field(res, grid)
	return func

def Cn_squared_from_fried_parameter(r0, wavelength):
	k = 2 * np.pi / wavelength
	return r0**(-5. / 3) / (0.423 * k**2)

def fried_parameter_from_Cn_squared(Cn_squared, wavelength):
	k = 2 * np.pi / wavelength
	return (0.423 * Cn_squared * k**2)**(-3. / 5)

def seeing_to_fried_parameter(seeing, wavelength):
	return 0.98 * wavelength / seeing

def fried_parameter_to_seeing(r0, wavelength):
	return 0.98 * wavelength / r0