import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats
import sys
import time
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from MLGW_generator import *
from GW_helper import * 	#routines for dealing with datasets

N_waves = 1
	#true waves
theta, true_amp, true_ph, times = create_dataset_TD(N_waves, N_grid = 2000,
                t_coal = .5, q_range = (1.,5.), m2_range = (10.,11.), s1_range = (-0.8,0.8), s2_range = (-0.,0.),
                t_step = 5e-5, lal_approximant = "SEOBNRv2_opt")

true_amp = true_amp*1e19
grad_amp = np.gradient(true_amp[0,:], 5e-5)

print(grad_amp)

#plt.plot(times, grad_amp, label = "grad")
plt.figure()
plt.plot(np.sign(times)*(np.abs(times)**0.35), true_amp[0,:], 'o', label = "amp scaled")

plt.figure()
plt.plot(times, true_amp[0,:], '-', label = "amp")
plt.legend()
plt.show()
