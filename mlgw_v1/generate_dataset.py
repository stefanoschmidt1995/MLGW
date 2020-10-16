###################
#	Wuick routine to generate a dataset of WFs
###################

from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *

if True:
	create_dataset_TD(20, N_grid = 2000, filename = "../datasets/GW_TD_mode_0/GW_dataset.dat",
                t_coal = 1.2, q_range = (1.,10.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8), #for full dataset
                t_step = 1e-4, approximant = "TEOBResumS", alpha = 0.5,
				path_TEOBResumS = '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/TEOBResumS/Python'
				)
#                t_coal = .05, q_range = (1.,5.), m2_range = None, s1_range = -0.3, s2_range = 0.2, #for s_const

#create_dataset_FD(5000, N_grid = 2048, filename = "../datasets/GW_std_dataset.dat",
#                q_range = (1.,5.), m2_range = 20., s1_range = (-0.85,0.85), s2_range = (-0.85,0.85),
#				log_space = True,
#                f_high = 200, f_step = .1/1600., f_max = None, f_min = 1, lal_approximant = "IMRPhenomPv2")



#quit()

theta_vector, amp_dataset, ph_dataset, x_grid = load_dataset("../datasets/GW_TD_mode_3/GW_TD_dataset.dat", shuffle=False, N_data = None) #loading
#print(theta_vector)

cut_off = -1
theta_vector= theta_vector[:, :cut_off]
amp_dataset = amp_dataset[:, :cut_off]
ph_dataset= ph_dataset[:, :cut_off]
x_grid=x_grid[:cut_off]

	#putting everything on a huge grid
#x_grid_huge = np.linspace(x_grid[0],x_grid[-1], 100000)
#N_huge = 3
#amp_huge = np.zeros((N_huge,len(x_grid_huge)))
#ph_huge = np.zeros((N_huge,len(x_grid_huge)))
#for i in range(N_huge):
#	amp_huge[i,:] = np.interp(x_grid_huge, x_grid, amp_dataset[i,:])
#	ph_huge[i,:] = np.interp(x_grid_huge, x_grid, ph_dataset[i,:])

plt.figure(0)
plt.title("Phase of dataset")
for i in range(20):
	plt.plot(x_grid, (ph_dataset[i,:]), label = str(i)+' '+str(theta_vector[i,0]))


plt.figure(1)
plt.title("Amplitude of dataset")
for i in range(20):
	plt.plot(x_grid, amp_dataset[i,:], label = str(i)+' '+str(theta_vector[i,0]))
#plt.legend()

plt.figure(2)
plt.title("Wave")
for i in range(20):
	plt.plot(x_grid, amp_dataset[i,:]*np.exp(1j*ph_dataset[i,:]).real, label = str(i)+' '+str(theta_vector[i,0]))

plt.show()
quit()

plt.figure(3)
plt.title("Feature vs q")
id_merger = np.where(x_grid ==0)[0]
print(id_merger)
plt.plot(theta_vector[:,0], ph_dataset[:,0], 'o', ms = 1)
plt.plot(theta_vector[:,0], ph_dataset[:,100], 'o', ms = 1)
plt.plot(theta_vector[:,0], ph_dataset[:,500], 'o', ms = 1)
plt.plot(theta_vector[:,0], ph_dataset[:,1000], 'o', ms = 1)
#plt.plot(theta_vector[:,0], ph_dataset[:,2500], 'o', ms = 1)
plt.plot(theta_vector[:,0], ph_dataset[:,id_merger-100], 'o', ms = 1)
plt.plot(theta_vector[:,0], ph_dataset[:,id_merger], 'o', ms = 1)
plt.plot(theta_vector[:,0], ph_dataset[:,-1], 'o', ms = 1)

plt.show()


quit()


