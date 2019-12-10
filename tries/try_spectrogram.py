###################
#	Some tries of fitting GW generation model using PCA + MoE
###################

import sys
#sys.path.insert(1, '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/routines')
sys.path.insert(1, '../routines')
from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy import signal

theta_vector, amp_dataset, ph_dataset, times = load_dataset("../datasets/GW_TD_dataset_small.dat", shuffle = False) #loading dataset
new_times = np.linspace(times[0],times[-1], len(times))

for i in range(amp_dataset.shape[0]):
	amp_dataset[i,:] = np.interp(new_times, times, 1e21*amp_dataset[i,:])
	ph_dataset[i,:] = np.interp(new_times, times, ph_dataset[i,:])

h = np.multiply(amp_dataset,np.exp(1j*ph_dataset))
new_times = new_times - new_times[0]
print(h[0,:].shape)

fs = 1./np.abs(new_times[0]-new_times[1])

f, t, Sxx = signal.spectrogram(h[0,:].real, fs, nperseg = 15000, noverlap = 25)
f_slice = np.where(f<3e4)

print(fs, t, new_times)

plt.pcolormesh(t, f[f_slice], np.abs(Sxx[f_slice]))
plt.colorbar()
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()




