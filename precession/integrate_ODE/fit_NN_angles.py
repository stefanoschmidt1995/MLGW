import numpy as np
import matplotlib.pyplot as plt

from precession_helper import *

import tensorflow as tf

	#creating dataset (if it is the case)
#create_dataset_alpha_beta(N_angles = 1000, filename = "starting_dataset.dat", N_grid = 500, tau_min = 10., q_range = (1.1,10.))
#quit()

ranges = np.array([(1.1,10.), (0.,1.), (0.,1.), (0., np.pi), (0., np.pi), (0., 2.*np.pi)])
dataset_generator = angle_generator(t_min = 10, N_times = 500, ranges = ranges, N_batch = 1000, replace_step = 20, load_file = "starting_dataset.dat")

def plot(model, folder):
	return plot_solution(model, 10, 10., 1995, folder = folder, show = False)

model = NN_precession("try_NN_angles")

model.load_everything("try_NN_angles/try_NN_angles")
#plot(model, ".")

model.fit(dataset_generator, N_epochs = 100,  learning_rate = 1e-3, save_output = True, plot_function = plot, checkpoint_step = 5, print_step = 1, validation_file = "validation_angles.dat")



