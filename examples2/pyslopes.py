from hcipy import *
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import sys

# Strictly speaking this isn't reconstruction, it's expressing a wavefront as a
# linear combination of basis elements. Names will be updated once reconstruction
# based on slope measurements is implemented.

N = eval(sys.argv[1])
xf = eval(sys.argv[2])
yf = eval(sys.argv[3])
if eval(sys.argv[4]) is None:
    show = True
else:
    show = eval(sys.argv[4])
sps = 40 * N //128

start = time.time()
basis_size = np.floor(4 * N/np.pi)
pyramid = PyramidWavefrontSensor(N, sps, 3.6e-3, 9.96, 1.625)
wf = Wavefront(pyramid.estimator.pupil_mask)
amplitude = 0.3
wf.electric_field = pupil_sin_phase(wf.electric_field, xf, yf, amplitude)
basis_fname = 'pydata' + os.sep + 'basis_' + str(N) + '_size_' + str(int(basis_size)) + '.dat'
if os.path.isfile(basis_fname):
    modal_basis_matrix = np.fromfile(basis_fname)
    modal_basis_matrix.shape = (N*N, modal_basis_matrix.size//(N*N))
else:
    modal_basis_matrix = pyramid.reconstructor.make_modal_basis(basis_size).transformation_matrix
    modal_basis_matrix.tofile(basis_fname)

prop_fname = 'pydata' + os.sep + 'prop' + str(N) + '_size_' + str(modal_basis_matrix.shape[-1])
if os.path.isfile(prop_fname):
    slopes_basis_matrix = np.fromfile(prop_fname)
else:
    flat = pyramid.optics.forward(Wavefront(pyramid.estimator.pupil_mask)).electric_field
    print("Starting the propagation of " + str(modal_basis_matrix.shape[-1]) + " basis elements.")
    slopes_basis = ModeBasis()
    for index, element in enumerate(modal_basis_matrix.T):
        if index % 100 == 0:
            print(index)
        slopes_basis.append(pyramid.optics.forward(Wavefront(Field(element, wf.electric_field.grid))).electric_field - flat)
    print("Done with propagations.")
    slopes_basis_matrix = slopes_basis.transformation_matrix
    slopes_basis_matrix.tofile(prop_fname)
slopes_basis_matrix.shape = (slopes_basis_matrix.size // modal_basis_matrix.shape[-1], modal_basis_matrix.shape[-1])

end = time.time()
if show:
    plt.subplot(2,2,1)
    imshow_field(wf.phase)
    x, y = pyramid.estimator.estimate(pyramid.propagate(wf))
    plt.subplot(2,2,2)
    imshow_field(pyramid.optics.forward(wf).intensity, vmin=0, vmax=0.5)
    plt.subplot(2,2,3)
    imshow_field(x, cmap='RdBu')
    plt.colorbar()
    plt.subplot(2,2,4)
    imshow_field(y, cmap='RdBu')
    plt.colorbar()
    plt.show()
    re = slopes_basis_matrix.dot(pyramid.reconstructor.get_weights(wf, modal_basis_matrix))
    s = int(np.sqrt(re.size // 2))
    showgrid = make_pupil_grid(s)
    '''re_x = Field(re[0:s*s] * circular_aperture(1)(showgrid), showgrid)
    re_y = Field(re[s*s:2*s*s] * circular_aperture(1)(showgrid), showgrid)
    plt.subplot(2,2,1)
    imshow_field(re_x)
    plt.colorbar()
    plt.subplot(2,2,2)
    imshow_field(re_y)
    plt.colorbar()
    plt.show()'''
# print("Time: " + str(np.round(end - start, 2)))
