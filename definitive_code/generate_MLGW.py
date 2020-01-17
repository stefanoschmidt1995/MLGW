######Wave generator!!!

###Alignment here is the major issue!! Aligning a wave at the beginning of time grid is the thing that seems t work better.
### However, this is not the cleanest thing one could do: the best would be aligning a wave at merger time...
### I think the least we fit the better it is!! So... we must use PN expansion up to the maximum we can reach. This is a guarantee that the fit will work at its best

###Of course we could as well try to improve our fitting method... But this is an uncertain path and for sure the previous things to try would help a lot despite the fitting method

import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats
import sys
import time
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from MLGW_generator import *
from GW_helper import * 	#routines for dealing with datasets

generator = MLGW_generator("TD", "./models_TD_long")

#testing performances
N_waves = 4

start_time = time.process_time_ns()/1e6 #ms

#theta_vector_test, amp_dataset_test, ph_dataset_test, frequencies_test = create_dataset_FD(N_waves, N_grid = 2048, filename = None,
#                q_range = (1.,5.), m2_range = 10, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
#				log_space = True,
#                f_high = 1000, f_step = 5e-2, f_max = None, f_min =20., lal_approximant = "IMRphenomPv2")
theta_vector_test, amp_dataset_test, ph_dataset_test, red_test_times = create_dataset_TD(N_waves, N_grid = int(200000), filename = None,
                t_coal = .4, q_range = (1.,5.), m2_range = (10.,10.), s1_range = (-0.8,0.6), s2_range = (-0.8,0.6),
				#t_coal = .015, q_range = 1., m2_range = (25.,25.0000001), s1_range = 0.8, s2_range = 0.6,
                t_step = 5e-5, lal_approximant = "SEOBNRv2_opt")


#amp_dataset_test, ph_dataset_test = generator.align_wave_TD(amp_dataset_test, ph_dataset_test, red_test_times, al_merger = True)

true_h = np.multiply(amp_dataset_test, np.exp(1j*ph_dataset_test))

middle_time = time.process_time_ns()/1e6

#"""
theta_vector_test = np.column_stack((theta_vector_test, np.full((N_waves,),1.), np.full((N_waves,),0.*np.pi)))
rec_amp_dataset, rec_ph_dataset = generator.get_WF(theta_vector_test, plus_cross = False, x_grid = red_test_times*20, red_grid = False)
rec_h = np.multiply(rec_amp_dataset, np.exp(1j*rec_ph_dataset))#"""

"""
rec_amp_dataset, rec_ph_dataset = generator(red_test_times*20., theta_vector_test[:,0], theta_vector_test[:,1],
            np.zeros((N_waves,)),np.zeros((N_waves,)), theta_vector_test[:,2],
            np.zeros((N_waves,)),np.zeros((N_waves,)), theta_vector_test[:,3],
            np.ones((N_waves,)), np.zeros((N_waves,)), np.full((N_waves,), 0.),
            np.zeros((N_waves,)),np.zeros((N_waves,)), np.zeros((N_waves,)), plus_cross = False )
rec_h = np.multiply(rec_amp_dataset, np.exp(1j*rec_ph_dataset))
#rec_h = h_plus+1j*h_cross
rec_amp_dataset = np.abs(rec_h)
rec_ph_dataset = np.unwrap(np.angle(rec_h))#"""

end_time = time.process_time_ns()/1e6

F = compute_mismatch(amp_dataset_test, ph_dataset_test, rec_amp_dataset, rec_ph_dataset)
print("Avg fit mismatch (avg,max,min,std): ", np.mean(F), np.max(F), np.min(F), np.std(F))

print("Time for lal (per WF): ", (middle_time-start_time)/float(N_waves), "ms\nTime for MLGW (per WF): ", (end_time-middle_time)/float(N_waves),"ms")
#print("Time for lal (per WF): ", (middle_time-start_time)/float(1), "ms\nTime for MLGW (per WF): ", (end_time-middle_time)/float(1),"ms")

#############PLOT TIME
	#plotting true and reconstructed waves	
to_plot = "h"

N_plots = 4
indices = np.random.choice(range(N_plots), size=N_plots ,replace = False)
for i in range(N_plots):
	plt.figure(i+1, figsize=(15,10))
#	m_tot = (theta_vector_test[indices[i],0]+1)*10.
	m_tot = (theta_vector_test[indices[i],0]+theta_vector_test[indices[i],1])
	#m_tot = 20. #if computation is not done on reduced grid
	plt.title("(q,s1,s2) = "+str(theta_vector_test[indices[i],:]))
	if to_plot == "h":
		plt.plot(red_test_times*20, rec_h[indices[i]].real, '-', label = "Rec")
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

