###################
#	Quick routine to generate a dataset of WFs
###################

try:
	from GW_helper import *
	from ML_routines import *
except:
	from mlgw.GW_helper import *
	from mlgw.ML_routines import *

try:
	import sys
	lm = sys.argv[1]
except:
	lm = "22" 	#mode to fit

if True:
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
	#The dataset is saved to a file, one file for each mode. The WF is time aligned s.t. the peak of the 22 mode happens at t = 0

	create_dataset_TD(26, N_grid = 2500, mode = (int(lm[0]),int(lm[1])), filename = "TD_datasets/{}_dataset.dat".format(lm),
                t_coal = 4., q_range = (1., 20.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
#      t_coal = 2., q_range = (1.,10.), m2_range = None, s1_range = (-1e-5,1e-5), s2_range = (-1e-5,1e-5), #for a nonspinning dataset
                t_step = 1e-4, alpha = 0.5,
				#approximant = "SEOBNRv2_opt"
				path_TEOBResumS = '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/TEOBResumS/Python'
				)

quit() #comment here to plot the dataset

############ Plotting dataset
import matplotlib.pyplot as plt

theta_vector, amp_dataset, ph_dataset, x_grid = load_dataset("TD_datasets/22_dataset_try.dat", shuffle=False, N_data = None) #loading the dataset

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



