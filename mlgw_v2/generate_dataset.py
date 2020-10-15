###################
#	Quick routine to generate a dataset of WFs
###################

from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *

if False:
	create_dataset_TD_TEOBResumS(4000, N_grid = 2000, mode = (4,4), filename = "TD_datasets/44_dataset.dat",
                t_coal = 2., q_range = (1.,10.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8), #for full dataset
                t_step = 1e-4, alpha = 0.5,
				path_TEOBResumS = '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/TEOBResumS/Python'
				)

if False:
	create_shift_dataset(6000, [(3,2),(3,3),(4,4)], filename = "TD_datasets/shift_dataset.dat",
				q_range = (1.,10.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
				path_TEOBResumS = '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/TEOBResumS/Python'
				)

#                t_coal = .05, q_range = (1.,5.), m2_range = None, s1_range = -0.3, s2_range = 0.2, #for s_const

quit()

#########Dealing with shifts
line_to_fit = 0
train_frac = 0.8

data = np.loadtxt("TD_datasets/shift_dataset.dat")

	#removing high spins
#ids = np.where(np.logical_and(data[:,1] < 0.8,data[:,2] < 0.8) )
#np.savetxt("TD_datasets/shift_dataset_ok.dat", data[ids], header = "# row: theta (3) | shifts (3) for modes [(3, 2), (3, 3), (4, 4)] \n # | q_range = (1.0, 10.0) | m2_range = None | s1_range = (-0.8, 0.8) | s2_range = (-0.8, 0.8) ")

train_data = data[:int(train_frac*data.shape[0]),:]
test_data = data[int(train_frac*data.shape[0]):,:]

train_theta = train_data[:,:3]
train_shifts = train_data[:,3+line_to_fit]
test_theta = test_data[:,:3]
test_shifts = test_data[:,3+line_to_fit]

plt.figure()
plt.scatter(train_theta[:,1], train_shifts, s = 1)

plt.show()


quit()
############ Plotting dataset


theta_vector, amp_dataset, ph_dataset, x_grid = load_dataset("TD_datasets/44_dataset.dat", shuffle=False, N_data = None) #loading
#print(theta_vector)

	#putting everything on a huge grid
#x_grid_huge = np.linspace(x_grid[0],x_grid[-1], 100000)
#N_huge = 3
#amp_huge = np.zeros((N_huge,len(x_grid_huge)))
#ph_huge = np.zeros((N_huge,len(x_grid_huge)))
#for i in range(N_huge):
#	amp_huge[i,:] = np.interp(x_grid_huge, x_grid, amp_dataset[i,:])
#	ph_huge[i,:] = np.interp(x_grid_huge, x_grid, ph_dataset[i,:])

N_plots = 50

plt.figure(0)
plt.title("Phase of dataset")
for i in range(N_plots):
	plt.plot(x_grid, (ph_dataset[i,:]), label = str(i)+' '+str(theta_vector[i,0]))


plt.figure(1)
plt.title("Amplitude of dataset")
for i in range(N_plots):
	plt.plot(x_grid, amp_dataset[i,:], label = str(i)+' '+str(theta_vector[i,0]))
#plt.legend()

plt.figure(2)
plt.title("Wave")
for i in range(N_plots):
	plt.plot(x_grid, amp_dataset[i,:]*np.exp(1j*ph_dataset[i,:]).real, label = str(i)+' '+str(theta_vector[i,0]))

plt.show()
quit()




