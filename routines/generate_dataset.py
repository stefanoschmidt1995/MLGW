###################
#	Some tries of fitting GW generation model using PCA+ logistic regression
#	Apparently it works quite well
###################

from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *

#create_dataset(10000, N_grid = 2048, filename = "../datasets/GW_std_dataset_enlarged.dat",
#                q_range = (.8,5.2), s1_range = (-0.85,0.85), s2_range = (-0.85,0.85),
#				log_space = True,
#                f_high = 1000, f_step = 5e-2, f_max = None, f_min =None, lal_approximant = "IMRPhenomPv2")

create_dataset(5000, N_grid = 2048, filename = "../datasets/GW_std_dataset.dat",
                q_range = (.9,5), m2_range = 20., s1_range = (-0.85,0.85), s2_range = (-0.85,0.85),
				log_space = True,
                f_high = 200, f_step = .1/1600., f_max = None, f_min = 1, lal_approximant = "IMRPhenomPv2")



quit()

theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("../datasets/GW_std_dataset.dat", shuffle=False, N_data = None) #loading

plt.figure(1)
plt.title("Amplitude of dataset")
for i in range(30):
	plt.plot(frequencies, amp_dataset[i,:], label = str(i)+' '+str(theta_vector[i,0]))
#plt.legend()

plt.figure(2)
plt.title("Phase of dataset")
for i in range(1):
	plt.plot(frequencies, amp_dataset[i,:]*np.exp(1j*ph_dataset[i,:]), label = str(i)+' '+str(theta_vector[i,0]))
#plt.legend()
plt.show()


print("Loaded data with shape: "+ str(ph_dataset.shape))


quit()


