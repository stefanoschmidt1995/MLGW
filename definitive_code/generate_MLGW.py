######Wave generator!!!

import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats
import sys
import time
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from MLGW_generator import *
from GW_helper import * 	#routines for dealing with datasets

generator = MLGW_generator("./models")

#testing performances
N_waves = 50

start_time = time.process_time_ns()/1e6 #ms

theta_vector_test, amp_dataset_test, ph_dataset_test, frequencies_test = create_dataset(N_waves, N_grid = 2048, filename = None,
                q_range = (1.,5.), s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
				log_space = True,
                f_high = 1000, f_step = 5e-2, f_max = None, f_min =None, lal_approximant = "IMRphenomPv2")
amp_dataset_test = 1e21*amp_dataset_test
true_h = np.multiply(amp_dataset_test, np.exp(1j*ph_dataset_test))

middle_time = time.process_time_ns()/1e6

rec_amp_dataset, rec_ph_dataset = generator.get_WF(theta_vector_test, plus_cross = False)#, freq_grid = frequencies_test)
h_plus = np.multiply(rec_amp_dataset, np.exp(1j*rec_ph_dataset))
#h_plus, h_cross =  generator.get_WF(theta_vector_test, freq_grid = frequencies_test)

end_time = time.process_time_ns()/1e6

F = compute_mismatch(amp_dataset_test, ph_dataset_test, rec_amp_dataset, rec_ph_dataset)
print("Avg fit mismatch (avg,max,min,std): ", np.mean(F), np.max(F), np.min(F), np.std(F))

print("Time for lal (per WF): ", (middle_time-start_time)/float(N_waves), "ms\nTime for MLGW (per WF): ", (end_time-middle_time)/float(N_waves),"ms")

#############PLOT TIME

	#plotting histogram for mismatch
plt.figure(0)
plt.title("Mismatch distributions:\n N_data = "+str(N_waves), fontsize = 15)
plt.hist(F*1e4, bins = 200)
plt.annotate("$\mu$ = "+'{:.2e}'.format(np.mean(F))+
			"\n$\sigma$ = "+ '{:.2e}'.format(np.std(F))+
			"\n$median$ = "+ '{:.2e}'.format(np.median(F)),
        xy=(.8, 0.7), xycoords='axes fraction', fontsize = 13,  horizontalalignment='left')
plt.annotate("$P(F<10^{-5})$ = "+'{:.2}'.format(len(np.where(F<1e-5)[0])/N_waves)+
			"\n$P(F<10^{-4})$ = "+'{:.2}'.format(len(np.where(F<1e-4)[0])/N_waves)+
			"\n$P(F<10^{-3})$ = "+'{:.2}'.format(len(np.where(F<1e-3)[0])/N_waves),
        xy=(.8, 0.5), xycoords='axes fraction', fontsize = 10,  horizontalalignment='left')
#plt.xscale("log")
#plt.yscale("log")
plt.xlabel("F (1e-4)")

	#plotting true and reconstructed waves	
N_plots = 3
indices = np.random.choice(range(N_plots), size=N_plots ,replace = False)
for i in range(N_plots):
	plt.figure(i+1, figsize=(15,10))
	plt.title("(q,s1,s2) = "+str(theta_vector_test[indices[i],0:3]))
	plt.plot(frequencies_test, h_plus[indices[i]], label = "Rec")
	plt.plot(frequencies_test, true_h[indices[i]].real, label = "True")
	plt.xscale("log")
	plt.legend()
	plt.savefig("../pictures/rec_WFs/WF_"+str(i)+".jpeg")

plt.show()

