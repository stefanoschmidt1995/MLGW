###################
#	Some tries of fitting GW generation model using PCA+ logistic regression
#	Apparently it works quite well
###################

from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *

if True:
	create_dataset_TD(5000, N_grid = 2000, filename = "../datasets/GW_TD_dataset_long.dat",
                t_coal = .4, q_range = (1.,5.), m2_range = 10., s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
                t_step = 5e-5, lal_approximant = "SEOBNRv2_opt")

#create_dataset_FD(5000, N_grid = 2048, filename = "../datasets/GW_std_dataset.dat",
#                q_range = (1.,5.), m2_range = 20., s1_range = (-0.85,0.85), s2_range = (-0.85,0.85),
#				log_space = True,
#                f_high = 200, f_step = .1/1600., f_max = None, f_min = 1, lal_approximant = "IMRPhenomPv2")



quit()

theta_vector, amp_dataset, ph_dataset, x_grid = load_dataset("../datasets/GW_TD_dataset_long.dat", shuffle=False, N_data = None) #loading

cut_off = 2500
theta_vector= theta_vector[:, :cut_off]
amp_dataset = amp_dataset[:, :cut_off]
ph_dataset= ph_dataset[:, :cut_off]
x_grid=x_grid[:cut_off]

	#putting everything on a huge grid
x_grid_huge = np.linspace(x_grid[0],x_grid[-1], 100000)
N_huge = 3
amp_huge = np.zeros((N_huge,len(x_grid_huge)))
ph_huge = np.zeros((N_huge,len(x_grid_huge)))
for i in range(N_huge):
	amp_huge[i,:] = np.interp(x_grid_huge, x_grid, amp_dataset[i,:])
	ph_huge[i,:] = np.interp(x_grid_huge, x_grid, ph_dataset[i,:])

plt.figure(0)
plt.title("Phase of dataset")
for i in range(10):
	plt.plot(x_grid, (ph_dataset[i,:]), label = str(i)+' '+str(theta_vector[i,0]))


plt.figure(1)
plt.title("Amplitude of dataset")
for i in range(30):
	plt.plot(x_grid, amp_dataset[i,:], label = str(i)+' '+str(theta_vector[i,0]))
#plt.legend()

plt.figure(2)
plt.title("Reconstructed wave")
for i in range(1):
	plt.plot(x_grid_huge, amp_huge[i,:]*np.exp(1j*ph_huge[i,:]), label = str(i)+' '+str(theta_vector[i,:]))
#plt.legend()
plt.show()


print("Loaded data with shape: "+ str(ph_dataset.shape))


quit()


