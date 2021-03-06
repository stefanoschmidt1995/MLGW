import numpy as np
import matplotlib.pyplot as plt

from precession_helper import *

import tensorflow as tf

#running on stefano.schmidt@ldas-pcdev11.ligo.caltech.edu

	#creating dataset (if it is the case)
#create_dataset_alpha_beta(N_angles = 2000, filename = "temp.dat", N_grid = 500, tau_min = 10., q_range = (1.1,10.), smooth_oscillation = True, verbose = True)
#quit()

ranges = np.array([(1.1,10.), (0.,1.), (0.,1.), (0., np.pi), (0., np.pi), (0., 2.*np.pi)])
dataset_generator = angle_generator(t_min = 10., N_times = 200, ranges = ranges, N_batch = 400, replace_step = None, load_file = "starting_dataset.dat", smooth_oscillation = True)

def plot(model, folder):
	return plot_validation_set(model, 10, "validation_angles.dat",folder = folder, show= False, smooth_oscillation = True)
	#return plot_solution(model, 10, 10., 1995, folder = folder, show = False, smooth_oscillation = True)

model = NN_precession("NN_smooth", smooth_oscillation = True)
model.summary()

#model.load_everything("NN_smooth/NN_smooth")
model.fit(dataset_generator, N_epochs = int(1e7),  learning_rate = 1e-4, save_output = True, plot_function = plot, checkpoint_step = 1000, print_step = 500, validation_file = "validation_angles.dat")

quit()
model.load_everything("NN_smooth/NN_smooth")
hist = np.array(model.history)
met = np.array(model.metric)
plt.plot(hist[:,0],hist[:,1])
plt.plot(met[:,0],met[:,1:])
plt.yscale('log')

plot_solution(model, 10, 10., 1995, folder = None, show = True)

plt.show()


