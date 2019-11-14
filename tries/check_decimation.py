###################
#	Some tries of fitting GW generation model using PCA + logistic regression
#	Apparently it works quite well
###################

from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
import scipy.interpolate

w_min = 0.05
w_max = 1.2
dw    = 0.0001

#create_dataset_red_f(20, N_grid=int((w_max-w_min)/dw), q_max = 1., spin_mag_max = 0., w_min =w_min, w_max = w_max, filename = "./datasets/GW_huge_Ngrid.dat") #for generating dataset from scratch
#f_vector = np.linspace(30,500,5000)
#theta_vector, w_amp_dataset, w_ph_dataset, w_vector = load_dataset("./datasets/GW_huge_Ngrid.dat", shuffle=False) #loading dataset
#f_amp_dataset, f_ph_dataset = transform_dataset(theta_vector, w_amp_dataset, w_ph_dataset, w_vector, f_vector, set_w_grid = False)

create_dataset(20, N_grid = None, filename = "./datasets/GW_huge_Ngrid.dat", q_max = 5, spin_mag_max = 0.8, f_high = 1000, f_step = 1e-2, f_max = None, f_min =None, lal_approximant = "IMRPhenomPv2")
w_vector = np.linspace(0.05,2,5000)
theta_vector, f_amp_dataset, f_ph_dataset, f_vector = load_dataset("./datasets/GW_huge_Ngrid.dat", shuffle=False, N_data = 10) #loading dataset
w_amp_dataset, w_ph_dataset = transform_dataset(theta_vector, f_amp_dataset, f_ph_dataset, w_vector, f_vector, set_w_grid = True)

print("Loaded "+ str(theta_vector.shape[0])+" data")




for i in range(10):
	plt.figure(0)
	plt.title('real f')
	plt.plot(f_vector, f_amp_dataset[i,:])
	plt.figure(1)
	plt.title('real f')
	plt.plot(f_vector, f_ph_dataset[i,:], label = str(theta_vector[i,:]))
	plt.legend()

for i in range(10):
	plt.figure(2)
	plt.title('red f')
	plt.plot(w_vector, w_amp_dataset[i,:])
	plt.figure(3)
	plt.title('red f')
	plt.plot(w_vector, w_ph_dataset[i,:])
plt.show()


quit()
N_grid_list = [100000, 200000]

for N_grid in N_grid_list:
	theta_vector_small, amp_dataset_small, ph_dataset_small, frequencies_small = load_dataset("./datasets/GW_huge_Ngrid.dat", shuffle=False, N_grid = N_grid)
	int_amp = np.zeros(amp_dataset.shape)
	int_ph =  np.zeros(ph_dataset.shape)
	sci_amp = np.zeros(ph_dataset.shape)
	sci_ph = np.zeros(ph_dataset.shape)

	for i in range(amp_dataset.shape[0]):
		amp_interp = scipy.interpolate.interp1d(frequencies_small, amp_dataset_small[i,:], kind='cubic', fill_value = 'extrapolate')
		ph_interp = scipy.interpolate.interp1d(frequencies_small, ph_dataset_small[i,:], kind='cubic', fill_value = 'extrapolate')
		sci_ph[i,:] = ph_interp(frequencies)
		sci_amp[i,:] = amp_interp(frequencies)

		int_amp[i,:] = np.interp(frequencies, frequencies_small, amp_dataset_small[i,:])
		int_ph[i,:] = np.interp(frequencies, frequencies_small, ph_dataset_small[i,:])

	F = compute_mismatch(amp_dataset, ph_dataset, int_amp, int_ph)
#print("Mismatch PCA: ",F_PCA)
	print("Mismatch decimation | N_grid, avg: ",N_grid, np.mean(F))
	F_sci = compute_mismatch(amp_dataset, ph_dataset, sci_amp, sci_ph)
#print("Mismatch PCA: ",F_PCA)
	print("Mismatch decimation sci | N_grid, avg: ",N_grid, np.mean(F_sci))

