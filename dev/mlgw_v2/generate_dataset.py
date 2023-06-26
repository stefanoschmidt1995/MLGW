###################
#	Quick routine to generate a dataset of WFs
###################

from GW_helper import *
from ML_routines import *

try:
	from GW_helper import *
	from ML_routines import *
except:
	from mlgw.GW_helper import *
	from mlgw.ML_routines import *

try:
	import sys
	lm_list = sys.argv[1:]
	for i, lm in enumerate(lm_list):
		lm_list[i] = (int(lm[0]),int(lm[1]))
	if lm_list == []:
		raise RuntimeError("Empty list")
except:
	lm_list = [(2,2)] 	#modes to fit

if False:
	#With create_dataset_TD_TEOBResumS a dataset of WF is created. The user must provide
	#	-the number of WFs to be generated
	#	-the number of grid points
	#	-the modes to generate [(l,m)]
	#	-a base filename to save the datasets at (each mode will be saved at basefilename.lm)
	#	-time to coalescence (in s/M_sun)
	#	-range of random parameters q, s1, s2 to generate the WFs with. If m2_range is None, a std total mass of 20 M_sun is used.
	#	-integration step for the EOB model
	#	-distortion parameter alpha for the time grid (in range (0,1)); a value of 0.3-0.5 is advised.
	#	-path to a local installation of TEOBResumS: it must have the module 'EOBRun_module'
	#The dataset is saved to a file, one file for each mode. The WF is time aligned s.t. the peak of the 22 mode happens at t = 0
	#Pay attention to the sampling rate!!! If it's too low, you will get very bad dataset

	create_dataset_TD(5400, N_grid = 2000, modes = lm_list, basefilename = "TD_datasets/IMRPhenomTPHM_dataset",
                t_coal = 2., q_range = (1., 10.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
#    	  t_coal = 2., q_range = (1.,10.), m2_range = None, s1_range = (-1e-5,1e-5), s2_range = (-1e-5,1e-5), #for a nonspinning dataset
                t_step = 1e-4, alpha = 0.5,
				approximant = "IMRPhenomTPHM",
				path_TEOBResumS = '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/TEOBResumS/Python'
				)

#quit() #comment here to plot the dataset

############ Plotting dataset
import matplotlib.pyplot as plt

N_plots = 10 #number of WFs to be plotted

theta_vector, amp_dataset, ph_dataset, x_grid = load_dataset("TD_datasets/IMRPhenomTPHM_dataset.21", shuffle=False, N_data = N_plots) #loading the dataset
#theta_vector, amp_dataset, ph_dataset, x_grid = load_dataset("TD_datasets/TEOB_dataset.21", shuffle=False, N_data = N_plots) #loading the dataset

N_trim = amp_dataset.shape[1]-30

plt.figure(0)
plt.title("Phase of dataset")
for i in range(N_plots):
	plt.plot(x_grid[:N_trim], (ph_dataset[i,:N_trim]), label = str(i)+' '+str(theta_vector[i,0]))


plt.figure(1)
plt.title("Amplitude of dataset")
for i in range(N_plots):
	plt.plot(x_grid[:N_trim], amp_dataset[i,:N_trim], 'o', ms = 2, label = str(i)+' '+str(theta_vector[i,0]))
#plt.legend()

plt.figure(2)
plt.title("Waveforms")
for i in range(N_plots):
	plt.plot(x_grid[:N_trim], amp_dataset[i,:N_trim]*np.exp(1j*ph_dataset[i,:N_trim]).real, label = str(i)+' '+str(theta_vector[i,0]))

plt.show()
quit()



