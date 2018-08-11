from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator, WavefrontSensorReconstructor, WavefrontSensor
from ..propagation import FraunhoferPropagator
from ..plotting import imshow_field
from ..optics import SurfaceApodizer, PhaseApodizer
from ..field import make_pupil_grid, make_focal_grid, Field
from ..aperture import circular_aperture
from ..mode_basis import ModeBasis, make_sine_basis
from ..optics import Wavefront
from ..math_util import least_inv

import numpy as np
from matplotlib import pyplot as plt # just for testing

def pyramid_surface(refractive_index, separation, wavelength_0):
	'''Creates a function which can create a pyramid surface on a grid.

	Parameters
	----------
	separation : scalar
		The separation of the pupils in pupil diameters.
	wavelength_0 : scalar
		The reference wavelength for the filter specifications.
	refractive_index : lambda function
		A lambda function for the refractive index which accepts a wavelength.

	Returns
	----------
	func : function
		The returned function acts on a grid to create the pyramid surface for that grid.

	'''
	def func(grid):
		surf = -separation / (refractive_index(wavelength_0) - 1) * (np.abs(grid.x) + np.abs(grid.y))
		return SurfaceApodizer(Field(surf, grid), refractive_index)
	return func

class PyramidWavefrontSensor(WavefrontSensor):
	'''The combined optics and estimator for a pyramid wavefront sensor. (Created to make input variables to Optics usable by Estimator, and possibly also vice versa.)

	Parameters
	----------
	pupil_size : scalar
		The side length (in pixels) of the input pupil grid.
	pupil_diameter : scalar
		The size of the pupil.
	grid_separation : scalar
		The separation distance between the pupils on the output grid in pupil diameters.
	grid_size : scalar
		The side length (in pixels) of the grid on which the output is sampled.
	grid_diameter : scalar
		The physical size of the output grid.
	subpupil_pixels : scalar
		The side length of a PWFS sub-image.
	aperture : function
		The telescope aperture on whose images the sensor is acting.
		Practically, acts as a mask for the input grid.
	wavelength_0 : scalar
		The reference wavelength which determines the physical scales.
		Passed directly to PWFSOptics constructor.
	q : scalar
		The focal plane oversampling coefficient.
		Passed directly to PWFSOptics constructor.
	refractive_index : function
		A function that returns the refractive index as function of wavelength.
		Passed directly to PWFSOptics constructor.
	num_airy : int
		The size of the intermediate focal plane grid that is used in terms of lambda/D at the reference wavelength.
		Passed directly to PWFSOptics constructor.

	Attributes
	----------
	pupil_grid : Grid
		The input pupil grid. (Get pupil_size from this)
	output_grid : Grid
		The output pupil grid. (Get grid_size from this)
	pupil_diameter : scalar
		As in parameters.
	grid_diameter : scalar
		As in parameters.
	subpupil_pixels : scalar
		As in parameters.
	grid_separation : scalar
		As in parameters.
	optics : PyramidWavefrontSensorOptics
		The object representing the optics for the sensor.
	estimator : PyramidWavefrontSensorEstimator
		The object representing the estimator for the sensor.
	reconstructor : PyramidWavefrontSensorReconstructor
		The object representing the reconstructor for the sensor.
	'''

	def __init__(self, pupil_size, subpupil_pixels, grid_diameter, pupil_diameter=1, grid_separation=1.5, grid_size=None, aperture=None, wavelength_0=1, q=4, refractive_index=lambda x : 1.5, num_airy=None):
		self.pupil_diameter = pupil_diameter
		self.grid_diameter = grid_diameter
		self.grid_separation = grid_separation
		self.subpupil_pixels = subpupil_pixels
		self.pupil_size = pupil_size
		self.grid_size = grid_size

		if grid_size is None:
			grid_size = self.pupil_size
		if aperture is None:
			aperture = circular_aperture(pupil_diameter)

		self.pupil_grid = make_pupil_grid(pupil_size, self.pupil_diameter)
		self.output_grid = make_pupil_grid(grid_size, self.grid_diameter)

		self.optics = PyramidWavefrontSensorOptics(self, wavelength_0, q, refractive_index, num_airy)
		self.estimator = PyramidWavefrontSensorEstimator(self, aperture, self.optics.output_grid)
		self.reconstructor = PyramidWavefrontSensorReconstructor(self)

	def propagate(self, wf):
		'''Utility method, returns the result of propagating a wavefront through the pyramid's optics.

		Parameters
		----------
		wf : Wavefront
			The input wavefront.

		Returns
		-------
		ndarray
			One-dimensional array containing all the sub-images resulting from a wavefront, concatenated.
		'''
		output = self.estimator.get_sub_images(self.optics.forward(wf).intensity)
		return output

class PyramidWavefrontSensorOptics(WavefrontSensorOptics):
	'''The optical elements for a pyramid wavefront sensor.

	Parameters
	----------
	sensor : PyramidWavefrontSensor
		The pyramid wavefront sensor from which attributes shared by this and Estimator are inherited.
	q : scalar
		As in PWFSensor.
	wavelength_0 : scalar
		As in PWFSensor.
	refractive_index : function
		As in PWFSensor.
	num_airy : int
		As in PWFSensor.

	Attributes
	----------
	subpupil_pixels : int
		The side length in pixels of one sub-image. Equal to num_pupil_pixels.
	output_grid : Grid
		The output grid of the wavefront sensor.
	focal_grid : Grid
		The intermediate focal plane grid where the focal plane is sampled.
	pupil_to_focal : FraunhoferPropagator
		A propagator for the input pupil plane to the intermediate focal plane.
	focal_to_pupil : FraunhoferPropagator
		A propagator for the intermediate focal plane to the output pupil plane.
	pyramid : SurfaceApodizer
		The filter that is applied in the focal plane.

	'''
	def __init__(self, sensor, wavelength_0, q, refractive_index, num_airy):
		# Removed the ability to have pupil_diameter of None which is corrected by referencing pupil_grid.x.ptp, since
		# the pupil grid is now made from the diameter (which defaults to 1) and the size in pixels.

		# Make mask
		sep = 0.5 * sensor.grid_separation * sensor.pupil_diameter

		# Multiply by 2 because we want to have two pupils next to each other
		output_grid_size = (sensor.grid_separation + 1) * sensor.pupil_diameter
		output_grid_pixels = np.ceil(sensor.subpupil_pixels * (sensor.grid_separation + 1))

		# Need at least two times over sampling in the focal plane because we want to separate two pupils completely
		if q < 2 * sensor.grid_separation:
			q = 2 * sensor.grid_separation

		# Create the intermediate and final grids
		self.output_grid = make_pupil_grid(output_grid_pixels, output_grid_size)
		self.focal_grid = make_focal_grid(sensor.pupil_grid, q=q, num_airy=num_airy, wavelength=wavelength_0)

		# Make all the optical elements
		self.pupil_to_focal = FraunhoferPropagator(sensor.pupil_grid, self.focal_grid, wavelength_0=wavelength_0)
		self.pyramid = pyramid_surface(refractive_index, sep, wavelength_0)(self.focal_grid)
		self.focal_to_pupil = FraunhoferPropagator(self.focal_grid, self.output_grid, wavelength_0=wavelength_0)

	def forward(self, wf):
		'''Propagates a wavefront through the pyramid wavefront sensor.

		Parameters
		----------
		wavefront : Wavefront
			The input wavefront that will propagate through the system.

		Returns
		-------
		wf : Wavefront
			The output wavefront.
		'''
		wf = self.pupil_to_focal.forward(wf)
		wf = self.pyramid.forward(wf)
		wf = self.focal_to_pupil(wf)

		return wf

class PyramidWavefrontSensorEstimator(WavefrontSensorEstimator):
	'''Estimates the wavefront slopes from pyramid wavefront sensor images.

	Parameters
	----------
	sensor : PyramidWavefrontSensor
		The pyramid from which images are being estimated.
	aperture : function
		A function which mask the pupils for the normalized differences.

	Attributes
	----------
	measurement_grid : Grid
		The grid on which the normalized differences are defined.
	pupil_mask : array_like
		A mask for the normalized differences.
	num_measurements : int
		The number of pixels in the output vector. (The size of the real vector representing the state of the estimated images.)
	'''
	def __init__(self, sensor, aperture, outgrid):
		Dsps = sensor.grid_diameter * sensor.subpupil_pixels / sensor.pupil_size
		self.subpupil_grid = make_pupil_grid(sensor.subpupil_pixels, Dsps)
		self.measurement_grid = outgrid
		self.pupil_mask = aperture(sensor.pupil_grid)
		self.grid_mask = aperture(self.subpupil_grid)
		self.num_measurements = 2 * int(np.sum(self.pupil_mask > 0))

	def get_sub_images(self, image):
		'''A function which extracts the sub-images from a WFS output image.
		Parameters
		----------
		image : Field
			The output (electric field) of a wavefront sensor.

		Returns
		-------
		sub_images : list
			A list of valid Fields, each representing a sub-image of the pyramid wavefront sensor.
		'''
		# TODO: Check that buffer space outside the grid of size (subpupil_pixels * (pupsep + 1)) is getting removed correctly.
		images = image
		pysize = int(np.sqrt(image.size))
		sps = int(np.sqrt(self.subpupil_grid.size))
		images.shape = (pysize, pysize)
		sub_images = [images[pysize-sps-1:pysize-1, 0:sps], images[pysize-sps-1:pysize-1, pysize-sps-1:pysize-1],
	                  images[0:sps, 0:sps], images[0:sps, pysize-sps-1:pysize-1]]
		for count, img in enumerate(sub_images):
			img = img.ravel()
			img.grid = self.subpupil_grid
			sub_images[count] = img
		return sub_images

	def estimate(self, images):
		'''A function which estimates the wavefront slope from a pyramid image.

		Parameters
		----------
		images - List
			A list of scalar intensity fields containing pyramid wavefront sensor images.

		Returns
		-------
		xslopes - Field
			A field representing slopes in the x direction.
		yslopes - Field
			A field representing slopes in the y direction.
		'''

		I_b = images[0]
		I_a = images[1]
		I_c = images[2]
		I_d = images[3]
		norm = I_a + I_b + I_c + I_d
		I_x = (I_a + I_b - I_c - I_d) / norm
		I_y = (I_a - I_b - I_c + I_d) / norm
		return Field(I_x, self.subpupil_grid), Field(I_y, self.subpupil_grid)

	def aperture_plot(data):
		'''A function to plot data of size self.num_measurements on self.pupil_mask.

		Parameters
		----------
		data - Field
			Any type of data (e.g. a Wavefront's electric field or phase) that has been converted to
			a field of size num_measurements. (Not a valid field)

		Returns
		-------
		toplot - Field
			A valid field that can be plotted with imshow_field.
		'''

		project_onto = self.measurement_grid.deepcopy()
		N = self.measurement_grid.shape[-1]
		project_onto.shape = (N, N)

		count, i, j = 0, 0, 0
		while count < self.num_measurements:
			if np.real(project_onto[i][j]) > 0:
				project_onto[i][j] = data[count]
				count += 1
			j += 1
			if j == N - 1:
				j = 0
				i += 1
		toplot = Field(project_onto.ravel(), self.measurement_grid)
		return toplot

class PyramidWavefrontSensorReconstructor(WavefrontSensorReconstructor):
	'''Reconstructs a wavefront based on PWFSEstimator slopes.

	Parameters
	----------
	sensor : PyramidWavefrontSensor
		The sensor based on which reconstruction is required.

	Attributes
	----------
	measurement_grid : Grid
		As in PWFSEstimator.

	pupil_mask : array_like
		As in PWFSEstimator.
	'''

	def __init__(self, sensor):
		self.measurement_grid = sensor.estimator.measurement_grid
		self.pupil_mask = sensor.estimator.pupil_mask

	def make_zonal_basis(self):
		'''Makes a basis for zonal reconstruction by applying a spanning set of
		one-pixel aberrations.

		Returns
		-------
		ModeBasis
			An orthogonalized basis for zonal aberrations.
		'''
		wfref = self.measurement_grid[self.pupil_mask > 0]
		N = self.measurement_grid.x.ptp()
		l = np.zeros((N, N), dtype=complex)
		basis = ModeBasis()
		for i in range(N):
			for j in range(N):
				for ab in [1+0j, 0+1j]:
					if np.complex(wfref[i][j]) != 0:
						l[i][j] = ab
						basis_element = Wavefront(Field(np.round(l.ravel(), 3) * self.pupil_grid, self.measurement_grid))
						basis.append(make_slopes(basis_element))
						l[i][j] = 0+0j
		return basis.orthogonalized

	def make_modal_basis(self, N):
		'''Makes a basis for modal reconstruction by applying a spanning set of
		sine aberrations.

		Returns
		-------
		ModeBasis
			An orthogonalized basis for modal aberrations.
		'''

		basis = make_sine_basis(self.measurement_grid, make_pupil_grid(N))
		basis = ModeBasis(x * self.pupil_mask for x in basis)
		return basis.orthogonalized
		return ModeBasis(np.around(x, 3) for x in basis.orthogonalized)

	def as_basis_sum(self, wf, basis_matrix):
		'''Reconstructs a wavefront as a linear combination of basis elements.

		Parameters
		----------
		wf : Wavefront
			The wavefront to reconstruct.
		basis_matrix : ndarray
			The transformation matrix of a basis generated by make_modal_basis
			or make_zonal_basis.

		Returns
		-------
		Wavefront
			The closest wavefront to wf that is a linear combination of basis elements.
		'''
		electric = Field(wf.electric_field.real + (basis_matrix.dot(self.get_weights(wf, basis_matrix)) * self.pupil_mask).imag*1j, wf.electric_field.grid)
		return Wavefront(electric)

	def get_weights(self, wf, basis_matrix):
		'''Gets the linear combination of basis elements that constitute the wavefront.

		Parameters
		----------
		wf : Wavefront
			The wavefront to reconstruct.
		basis_matrix : ndarray
			The transformation matrix of a basis generated by make_modal_basis
			or make_zonal_basis.
		R : ndarray, optional
			The reconstructor matrix returned by make_reconstructor_matrix.

		Returns
		-------
		weights : ndarray
			The coefficients to be multiplied into the basis_matrix to make the wavefront.
		'''
		return basis_matrix.T.dot(wf.electric_field)
