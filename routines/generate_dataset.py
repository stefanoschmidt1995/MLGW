###################
#	Some tries of fitting GW generation model using PCA+ logistic regression
#	Apparently it works quite well
###################

from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *

create_dataset(5000, N_grid = 2048, filename = "../datasets/GW_std_dataset_s0.dat", q_max = 5, spin_mag_max = 0., f_high = 1000, f_step = 5e-2, f_max = None, f_min =None, lal_approximant = "IMRPhenomPv2")
quit()
theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("../datasets/GW_std_temp.dat", shuffle=False, N_data = None) #loading

plt.figure(1)
plt.title("Amplitude of dataset")
for i in range(300):
	plt.plot(frequencies, amp_dataset[i,:], label = str(i)+' '+str(theta_vector[i,0]))
#plt.legend()

plt.figure(2)
plt.title("Phase of dataset")
for i in range(300):
	plt.plot(frequencies, ph_dataset[i,:], label = str(i)+' '+str(theta_vector[i,0]))
#plt.legend()
plt.show()


print("Loaded data with shape: "+ str(ph_dataset.shape))


quit()


