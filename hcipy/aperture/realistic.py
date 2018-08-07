import numpy as np
from ..field import CartesianGrid, UnstructuredCoords, evaluate_supersampled
from .generic import make_obstructed_circular_aperture, make_spider

def make_vlt_aperture():
	pass

def make_subaru_aperture():
	pass

def make_lbt_aperture():
	pass

def make_magellan_aperture(normalized=False):
	'''Make the Magellan aperture.

	Parameters
	----------
	normalized : boolean
		If this is True, the outer diameter will be scaled to 1. Otherwise, the
		diameter of the pupil will be 6.5 meters.

	Returns
	-------
	Field generator
		The Magellan aperture.
	'''
	pupil_diameter = 6.5 #m
	spider_width1 = 0.75 * 0.0254 #m
	spider_width2 = 1.5 * 0.0254 #m
	central_obscuration_ratio = 0.29
	spider_offset = [0,0.34] #m

	if normalized:
		spider_width1 /= pupil_diameter
		spider_width2 /= pupil_diameter
		spider_offset = [x / pupil_diameter for x in spider_offset]
		pupil_diameter = 1

	spider_offset = np.array(spider_offset)

	mirror_edge1 = (pupil_diameter / (2*np.sqrt(2)), pupil_diameter / (2*np.sqrt(2)))
	mirror_edge2 = (-pupil_diameter / (2*np.sqrt(2)), pupil_diameter / (2*np.sqrt(2)))
	mirror_edge3 = (pupil_diameter / (2*np.sqrt(2)), -pupil_diameter / (2*np.sqrt(2)))
	mirror_edge4 = (-pupil_diameter / (2*np.sqrt(2)), -pupil_diameter / (2*np.sqrt(2)))

	obstructed_aperture = make_obstructed_circular_aperture(pupil_diameter, central_obscuration_ratio)
	spider1 =  make_spider(spider_offset, mirror_edge1, spider_width1)
	spider2 =  make_spider(spider_offset, mirror_edge2, spider_width1)
	spider3 =  make_spider(-spider_offset, mirror_edge3, spider_width2)
	spider4 =  make_spider(-spider_offset, mirror_edge4, spider_width2)

	def func(grid):
		return obstructed_aperture(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
	return func

def make_keck_aperture(normalized=False):
	'''Make the Keck aperture.
	To be implemented later: change aperture from circular to a set of hexagons

	Parameters
	----------
	normalized : boolean
		If this is True, the outer diameter will be scaled to 1. Otherwise, the
		diameter of the pupil will be 9.96 meters.

	Returns
	-------
	Field generator
		The Keck aperture.
	'''
	D_keck = 9.96 #m
	spider_width = 0.0254 #m
	central_obscuration_ratio = 0.266
	# Source for figures: http://www.oir.caltech.edu/twiki_oir/pub/Keck/NGAO/NotesKeckPSF/KeckPupil_Notes.png
	if normalized:
		spider_width /= D_keck
		D_keck = 1
	keck_aperture = make_obstructed_circular_aperture(D_keck, central_obscuration_ratio, 6, spider_width)
	def func(grid):
		return evaluate_supersampled(keck_aperture, grid, 8)
	return func

def make_luvoir_aperture():
	pass

def make_elt_aperture():
	pass
