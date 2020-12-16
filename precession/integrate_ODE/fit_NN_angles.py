import numpy as np
import matplotlib.pyplot as plt

from precession_helper import *

import tensorflow as tf


ranges = np.array([(1.1,10.), (0.,1.), (0.,1.), (0., np.pi), (0., np.pi), (0., 2.*np.pi)])
dataset_generator = angle_generator(10, 100, ranges = ranges, N_batch = 10, replace_step = 20)

model = NN_precession("try_NN_angles__")

model.fit(dataset_generator, N_epochs = 100,  learning_rate = 1e-3, save_output = True, plot_function = None, save_step = 50, print_step = 1)



