from hcipy import *
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import sys

N = eval(sys.argv[1])
xf = eval(sys.argv[2])
yf = eval(sys.argv[3])
if eval(sys.argv[4]) is None:
    show = True
else:
    show = eval(sys.argv[4])

start = time.time()
basis_size = np.floor(4 * N/np.pi)
pyramid = PyramidWavefrontSensor(N, 40 * N // 128, pupil_diameter=9.96, pupil_separation=1.625)
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
reconstructed_wf = pyramid.reconstructor.as_basis_sum(wf, modal_basis_matrix)
end = time.time()
residual = reconstructed_wf.phase - wf.phase
if show:
    plt.subplot(2,2,1)
    imshow_field(wf.phase)
    plt.title("Original")
    plt.colorbar()
    plt.subplot(2,2,2)
    imshow_field(reconstructed_wf.phase)
    plt.title("Reconstructed")
    plt.colorbar()
    plt.subplot(2,2,3)
    imshow_field(residual)
    plt.title("Residual")
    plt.colorbar()
    plt.show()
print("Time: " + str(np.round(end - start, 2)))
print("Residual mean: " + str(np.round(np.mean(np.real(residual)), 3)) + "+" + str(np.round(np.mean(np.imag(residual)), 3)) + "j")
print("Residual max: " + str(np.round(np.max(np.absolute(np.real(residual))), 3)) + "+" + str(np.round(np.max(np.absolute(np.imag(residual))), 3)) + "j")
max_index = np.unravel_index(np.argmax(np.absolute(residual), axis=None), residual.shape)
print("At index of max residual, original electric field is " + str(wf.electric_field[max_index]) + " and that of the reconstructed wavefront is " + str(reconstructed_wf.electric_field[max_index]))
