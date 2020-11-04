###################
#	Quick routine to generate a dataset of WFs
###################

try:
	from GW_helper import *
	from ML_routines import *
except:
	from mlgw.GW_helper import *
	from mlgw.ML_routines import *

if False:
	#With create_dataset_TD_TEOBResumS a dataset of WF is created. The user must provide
	#	-the number of WFs to be generated
	#	-the number of grid points
	#	-the mode to be generated (l,m)
	#	-an optional filename to save the dataset at (if None, dataset is returned to the user)
	#	-time to coalescence (in s/M_sun)
	#	-range of random parameters q, s1, s2 to generate the WFs with. If m2_range is None, a std total mass of 20 M_sun is used.
	#	-integration step for the EOB model
	#	-distortion parameter alpha for the time grid (in range (0,1)); a value of 0.3-0.5 is advised.
	#	-path to a local installation of TEOBResumS: it must have the module 'EOBRun_module'
	#The dataset is saved to a file, one file for each mode. The WF is time aligned s.t. the peak of each mode happens at t = 0

	create_dataset_TD_TEOBResumS(3000, N_grid = 2000, mode = (3,3), filename = "TD_datasets/33_dataset.dat",
                t_coal = 2., q_range = (1.,10.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
#      t_coal = 2., q_range = (1.,10.), m2_range = None, s1_range = (-1e-5,1e-5), s2_range = (-1e-5,1e-5), #for a nonspinning dataset
                t_step = 1e-4, alpha = 0.5,
				path_TEOBResumS = '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/TEOBResumS/Python'
				)

if False:
	#This is to build a 'shift dataset' crucial for the correct alignments of the modes.
	#Each mode is stored in the dataset s.t. its peak happens at t=0. This requires the ability of shifting properly each mode in order to have it aligned with the others.
	#Here shift refers to the difference (in reduced grid) between the peak of the lm mode and the peak of the 22 mode, with the same parameters.
	#Knowing the shifts ensures that the WF can be faithfuly reconstructed with all the modes correctly aligned.

	create_shift_dataset(7000, [(2,1), (3,1), (3,2), (3,3),(4,1),(4,2),(4,3),(4,4), (5,5)], filename = "TD_datasets/shift_dataset.dat",
				q_range = (1.,10.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
				path_TEOBResumS = '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/TEOBResumS/Python'
				)


quit() #comment here to plot the dataset

############ Plotting dataset
import matplotlib.pyplot as plt

theta_vector, amp_dataset, ph_dataset, x_grid = load_dataset("TD_datasets/33_dataset.dat", shuffle=False, N_data = None) #loading the dataset

N_plots = 50 #number of WFs to be plotted

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
plt.title("Waveforms")
for i in range(N_plots):
	plt.plot(x_grid, amp_dataset[i,:]*np.exp(1j*ph_dataset[i,:]).real, label = str(i)+' '+str(theta_vector[i,0]))

plt.show()
quit()

##############

line_to_fit = 0
train_frac = 0.8

data = np.loadtxt("TD_datasets/shift_dataset.dat")

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



