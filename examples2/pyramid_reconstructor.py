from hcipy import *
from matplotlib import pyplot as plt
import numpy as np

# For now this is a test but once it's done it'll be an example.
# Strictly speaking this isn't reconstruction, it's expressing a wavefront as a
# linear combination of basis elements. Names will be updated once reconstruction
# based on slope measurements is implemented.
pyramid = PyramidWavefrontSensor(128, 40, 9.96, 1.625)
modal_basis = pyramid.reconstructor.make_modal_basis()
wf = Wavefront(pyramid.estimator.pupil_mask)
amplitude = 0.3
spatial_frequency = 5
wf.electric_field *= np.exp(1j * amplitude * np.sin(2*np.pi * pyramid.pupil_grid.x / pyramid.pupil_diameter * spatial_frequency))
reconstructed_wf = pyramid.reconstructor.as_basis_sum(wf, modal_basis)
imshow_field(wf.phase)
plt.show()
imshow_field(reconstructed_wf.phase)
plt.show()
imshow_field(wf.electric_field)
plt.show()
imshow_field(reconstructed_wf.electric_field)
plt.show()
