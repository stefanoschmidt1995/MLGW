######Wave generator!!!

import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats
import time

from GW_generator import *
from GW_helper import * 	#routines for dealing with datasets

generator = GW_generator("./TD_model_TEOBResumS")
#generator = GW_generator(1)

generator.summary()

#testing performances
N_waves = 16
plot = True

m1_range = (5.,50.)
m2_range = (5.,50.)
s1_range = (-0.8,0.8)
s2_range = (-0.8,0.8)
d_range = (.5,10.)
i_range = (0,2*np.pi)
phi_0_range = (0,np.pi)

low_list = [m1_range[0],m2_range[0], s1_range[0], s2_range[0], d_range[0], i_range[0], phi_0_range[0]]
high_list = [m1_range[1],m2_range[1], s1_range[1], s2_range[1], d_range[1], i_range[1], phi_0_range[1]]

theta = np.random.uniform(low = low_list, high = high_list, size = (N_waves, 7))
F = np.zeros((N_waves,))
time_lal = np.zeros((N_waves,))
time_mlgw = np.zeros((N_waves,))

for i in range(N_waves):
	start_time = time.process_time_ns()/1e6 #ms
	times, h_p_true, h_c_true = generate_waveform(*theta[i,:], t_coal = 0.25, t_step = 1e-5)
	h = (h_p_true+1j*h_c_true)
	middle_time = time.process_time_ns()/1e6
	h_p, h_c = generator.get_WF(theta[i,:], out_type = 'h+x', t_grid = times , red_grid = False)
	h_rec = (h_p+1j*h_c)
	end_time = time.process_time_ns()/1e6
	F[i], phi_ref = compute_optimal_mismatch(h, h_rec)
	time_lal[i] = middle_time-start_time
	time_mlgw[i] = end_time-middle_time
	print("F: ",F[i])
	if plot:
		plt.figure(0)
		plt.title("(m1,m2,s1,s2,d, i, phi_0) = "+str(theta[i,:])+"\n F = "+str(F[i]) )
		plt.plot(times, h.real, '-', label = "true")
		plt.plot(times, (h_rec*np.exp(1j*phi_ref)).real, '-', label = "rec")
		plt.legend()
		plt.show()

start_full = time.process_time_ns()/1e6
generator.get_WF(theta, out_type = 'h+x', t_grid = times , red_grid = False)
end_full =  time.process_time_ns()/1e6

#"""
print("Avg fit mismatch (avg,max,min,std): ", np.mean(F), np.max(F), np.min(F), np.std(F))

print("Time for lal (per WF) (avg,max,min,std): ", np.mean(time_lal), np.max(time_lal), np.min(time_lal), np.std(time_lal),
"ms\nTime for MLGW (per WF) (avg,max,min,std): ", np.mean(time_mlgw), np.max(time_mlgw), np.min(time_mlgw), np.std(time_mlgw),"ms")
print("Parallel execution: ", (end_full-start_full)/N_waves)
quit()

###################################OLD CODE
#############PLOT TIME
	#plotting true and reconstructed waves	
to_plot = "h"

N_plots = 4
indices = np.random.choice(range(N_plots), size=N_plots ,replace = False)
for i in range(N_plots):
		#m_tot for the test wave 
	#m_tot = (theta_vector_test[indices[i],0]+theta_vector_test[indices[i],1])
	m_tot = 20. #if computation is not done on reduced grid

	plt.figure(i+1, figsize=(15,10))
	plt.title("(q,s1,s2) = "+str(theta_vector_test[indices[i],:]))
	if to_plot == "h":
		plt.plot(red_test_times*20., rec_h[indices[i]].real, '-', label = "Rec")
		plt.plot(red_test_times*m_tot, true_h[indices[i]].real, '-', label = "True")

	if to_plot == "ph":
		plt.plot(red_test_times*20, rec_ph_dataset[indices[i]], '-', label = "Rec")
		plt.plot(red_test_times*m_tot, ph_dataset_test[indices[i]], '-', label = "True")

	if to_plot == "amp":
		plt.plot(red_test_times*20, rec_amp_dataset[indices[i]], '-', label = "Rec")
		plt.plot(red_test_times*m_tot, amp_dataset_test[indices[i]], '-', label = "True")
	
	plt.legend()
	plt.savefig("../pictures/rec_WFs/WF_"+str(i)+".jpeg")
plt.show()

	#plotting histogram for mismatch
plt.figure(0)
plt.title("Mismatch distributions:\n N_data = "+str(N_waves), fontsize = 15)
plt.hist(F*1e3, bins = 200)
plt.annotate("$\mu$ = "+'{:.2e}'.format(np.mean(F))+
			"\n$\sigma$ = "+ '{:.2e}'.format(np.std(F))+
			"\n$median$ = "+ '{:.2e}'.format(np.median(F)),
        xy=(.8, 0.7), xycoords='axes fraction', fontsize = 13,  horizontalalignment='left')
plt.annotate("$P(F<10^{-5})$ = "+'{:.2}'.format(len(np.where(F<1e-5)[0])/N_waves)+
			"\n$P(F<10^{-4})$ = "+'{:.2}'.format(len(np.where(F<1e-4)[0])/N_waves)+
			"\n$P(F<10^{-3})$ = "+'{:.2}'.format(len(np.where(F<1e-3)[0])/N_waves)+
			"\n$P(F<10^{-2})$ = "+'{:.2}'.format(len(np.where(F<1e-2)[0])/N_waves)+
			"\n$P(F<10^{-1})$ = "+'{:.2}'.format(len(np.where(F<1e-1)[0])/N_waves),
        xy=(.8, 0.5), xycoords='axes fraction', fontsize = 10,  horizontalalignment='left')
#plt.xscale("log")
#plt.yscale("log")
plt.xlabel("F (1e-3)")



plt.show()

