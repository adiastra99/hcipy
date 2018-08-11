import numpy as np

from ..field import Field
from ..statistics import SpectralNoiseFactoryFFT
from .apodization import SurfaceApodizer
from ..propagation import FresnelPropagator
from ..mode_basis import make_zernike_basis
from ..math_util import inverse_tikhonov
from ..aperture import circular_aperture
from .optical_element import OpticalElement

def pupil_sin_phase(pupil, wavsx=1, wavsy=0, amplitude=0.1):
	'''Generates a sine aberration from which to make a modal reconstruction basis.

	Parameters
	---------
	pupil : Field
		The attribute of the input (sampled over the pupil grid of a WFS) to aberrate.
		Usually the electric_field attribute of a Wavefront object.
	wavsx : scalar
		The number of complete sine aberrations to apply across the pupil in the x direction.
	wavsy : scalar
		The number of complete sine aberrations to apply across the pupil in the y direction.
	amplitude : scalar
		The amplitude of the sine aberrations to apply. Should be less than/equal to 1.

	Returns
	-------
	Field
		Same type of physical object as pupil, with aberration applied.
	'''
	size = int(np.sqrt(pupil.size))
	x = np.arange(size)
	y = np.arange(size)
	sin = np.zeros((size,size))
	tau = 2 * np.pi

	try:
		angfreq_x = size / wavsx
		xfreq = tau / angfreq_x
	except ZeroDivisionError:
		xfreq = 0

	try:
		angfreq_y = size / wavsy
		yfreq = tau / angfreq_y
	except ZeroDivisionError:
		yfreq = 0

	for i in range(len(x)):
		for j in range(len(y)):
			sin[i,j] = amplitude * np.sin(xfreq * i + yfreq * j)

	return pupil * np.exp(complex(0,1)*sin).ravel()

def make_power_law_error(pupil_grid, ptv, diameter, exponent=-2.5, aperture=None, remove_modes=None):
	def psd(grid):
		res = Field(grid.as_('polar').r**-exponent, grid)
		res[grid.as_('polar').r == 0] = 0
		return res

	if aperture is None:
		aperture = circular_aperture(diameter)(pupil_grid)

	screen = SpectralNoiseFactoryFFT(psd, pupil_grid).make_random()()
	screen *= ptv / np.ptp(screen[aperture != 0])

	if remove_modes is not None:
		modes = make_zernike_basis(remove_modes, diameter, pupil_grid)
		trans = modes.transformation_matrix
		trans_inv = inverse_tikhonov(trans, 1e-6)
		screen -= trans.dot(trans_inv.dot(screen))

	return Field(np.exp(2j*np.pi * screen) * aperture, pupil_grid)

class SurfaceAberration(SurfaceApodizer):
	def __init__(self, pupil_grid, ptv, diameter, exponent=-2.5, refractive_index=-1, aperture=None, remove_modes=None):
		surface = make_power_law_error(pupil_grid, ptv, diameter, exponent, aperture, remove_modes)
		SurfaceApodizer.__init__(self, surface, refractive_index)

class SurfaceAberrationAtDistance(OpticalElement):
	def __init__(self, surface_aberration, distance):
		self.fresnel = FresnelPropagator(surface_aberration.input_grid, distance)
		self.surface_aberration = surface_aberration

	def forward(self, wavefront):
		wf = self.fresnel.forward(wavefront)
		wf = self.surface_aberration.forward(wf)
		return self.fresnel.backward(wf)

	def backward(self, wavefront):
		wf = self.fresnel.forward(wavefront)
		wf = self.surface_aberration.backward(wf)
		return self.fresnel.backward(wf)
