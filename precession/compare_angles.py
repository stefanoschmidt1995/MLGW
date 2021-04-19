import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../mlgw_v2')

import GW_generator as gen
from GW_helper import compute_optimal_mismatch, locate_peak

import lal
import lalsimulation as lalsim

from precession_helper import *

import scipy.optimize, scipy.integrate, scipy.interpolate

import lalintegrate_PNeqs

theta = np.array([.66666667, .33333333,-0.43,.4,-0.2, 0.5,0.,0.3])
theta_modes = np.concatenate([theta[:2],theta[[4,7]]]) #(N,4) #theta for generating the non-precessing WFs


g = gen.GW_generator(1)


#t_grid, alpha_bis, beta_bis = get_alpha_beta_M(theta[0]+theta[1], *g.get_precessing_params(theta[None,0],theta[None,1], theta[2:5],theta[5:8]), f_ref = 400., smooth_oscillation = False, verbose = True)

t_grid, alpha, beta = get_alpha_beta_L0frame(theta[0]+theta[1], theta[0]/theta[1], theta[2:5],theta[5:8], 400.)

t_grid_lal, alpha_lal, beta_lal = lalintegrate_PNeqs.get_alpha_beta(theta[0]/theta[1], theta[2:5],theta[5:8], 400., times = None, 
		t_shift = -0.,
		f_merger =g.get_merger_frequency(theta_modes)
		)

plt.figure()		
plt.plot(t_grid_lal, alpha_lal, label = 'lal')
plt.plot(t_grid, alpha, label = 'precession package')
plt.legend()
		
plt.figure()		
plt.plot(t_grid_lal, beta_lal, label = 'lal')
plt.plot(t_grid, beta, label = 'precession package')
plt.legend()

plt.show()
