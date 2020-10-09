import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import numpy as np
import scipy.stats
import sys
import time
import os

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)	

from GW_generator import *
from GW_helper import * 	#routines for dealing with datasets

modes_to_k = lambda modes:[int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes] # [(l,m)] -> [k]

load = False
plot = False

N_waves = 2000
filename = "accuracy/mismatch_hist_TEOBResumS.dat"

f_min = 10

np.random.seed(23)

modes = [(2,2), (3,3), (3,2), (4,4)]

if not load:

	F = np.zeros((N_waves,len(modes)+1))  #[F_tot, [F_modes]]
	generator = GW_generator("TD_models/model_0")

		#getting random theta
	m1_range = (5.,100.)
	m2_range = (5.,100.)
	s1_range = (-0.8,0.95)
	s2_range = (-0.8,0.95)
	d_range = (.5,10.)
	i_range = (0,0) 
	phi_0_range = (0,0)

	low_list = [m1_range[0],m2_range[0], s1_range[0], s2_range[0], d_range[0], i_range[0], phi_0_range[0]]
	high_list = [m1_range[1],m2_range[1], s1_range[1], s2_range[1], d_range[1], i_range[1], phi_0_range[1]]

	theta = np.random.uniform(low = low_list, high = high_list, size = (N_waves, 7))

	for i in range(N_waves):
		#times, h_p, h_c = generate_waveform(*theta[i,:], f_min = f_min, t_step = 1e-5, approx = "SEOBNRv2_opt")
		times, h_p, h_c, hlm, t_m = generate_waveform_TEOBResumS(*theta[i,:-1], f_min = f_min,
								verbose = False, t_step = 1e-4, modes = modes)

		h_p_rec, h_c_rec = generator.get_WF(theta[i,:],times)
		#amp_aligned, ph_aligned = generator.get_modes(theta[i,:],times, out_type = "ampph", modes = modes, align22 = True)
		#shifts = generator.get_shifts(theta[i,:], modes)

		for j in range(len(modes)):
			k = modes_to_k([modes[j]])
			#print("Times: ", times[np.argmax(np.abs(hlm[str(k[0])][0]))], shifts*(theta[i,0]+theta[i,1]))

			t_lm = times - times[np.argmax(np.abs(hlm[str(k[0])][0]))]
			amp, ph = generator.get_modes(theta[i,:], t_lm, out_type = "ampph", modes = modes[j], align22 = False)

			h = amp * np.exp(1j*ph)
			h_TEOB = hlm[str(k[0])][0] * np.exp(1j * hlm[str(k[0])][1])
			F_temp, phi_0 = compute_optimal_mismatch(h,h_TEOB) #things are aligned up to merger...
			print(i,modes[j],theta[i,:],F_temp)
			F[i,j+1] = F_temp

			if plot:			
				plt.figure()
				plt.title("Amp mode {}".format(str(modes[j])))
				plt.plot(t_lm, amp, label = "mlgw")
				#plt.plot(t_lm, amp_aligned[:,j], label = "mlgw - aligned")	
				plt.plot(t_lm, hlm[str(k[0])][0], label = "TEOB")
				plt.legend()
			
				plt.figure()
				plt.title("Ph mode {}".format(str(modes[j])))
				plt.plot(t_lm, ph, label = "mlgw")
				#plt.plot(t_lm, ph_aligned[:,j], label = "mlgw - aligned")	
				plt.plot(t_lm, hlm[str(k[0])][1], label = "TEOB")
				plt.legend()

				plt.show()


	f=open(filename,'ab') #to append
	if os.stat(filename).st_size == 0:
		np.savetxt(f,F, header = str(modes)) #putting header
	else:
		np.savetxt(f,F)
	f.close()


F = np.loadtxt(filename) #(N, len(modes)+1)
print("Loading histogram from: {}\n\t{} datapoints".format(filename, F.shape[0]))

fig, ax_list = plt.subplots(nrows = len(modes), ncols = 1, sharex = True)
plt.subplots_adjust(hspace = .8)
plt.suptitle("HMs mismatch", fontsize = 15)
ax_list[-1].set_xlabel(r"$\log \; \bar{\mathcal{F}}$")

for i in range(len(modes)):
	ax_list[i].hist(np.log10(F[:,i+1]), bins = 70, color = 'k', density = True)
	ax_list[i].set_title("Mode {}".format(modes[i]), fontsize = 12)

#for i in range(len(modes)-1):
#	ax_list[i].set_xticks(ax_list[-1].get_xticks())
#	ax_list[i].set_xticklabels(ax_list[-1].get_xticklabels())
#	for tk in ax_list[i].get_xticklabels():
#		tk.set_visible(True)


plt.savefig("accuracy/mismatch_HMs.pdf", format = 'pdf')


plt.show()














