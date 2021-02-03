import numpy as np
import matplotlib.pyplot as plt
import scipy.signal, scipy.interpolate

from precession_helper import *

import sys
sys.path.insert(0,'../../mlgw_v2')
from GW_helper import *
from ML_routines import *

ranges = np.array([(1.1,10.), (0.,1.), (0.,1.), (0., np.pi), (0., np.pi), (0., 2.*np.pi)])
dataset_generator = angle_generator(t_min = 10., N_times = 200, ranges = ranges, N_batch = 2000, replace_step = None, load_file = "starting_dataset.dat", smooth_oscillation = True)

for X in dataset_generator():
	print(X.shape)
	
	plt.plot(X[:,0], X[:,8], 'o', ms = 2)
	plt.show()
