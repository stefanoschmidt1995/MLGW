import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import numpy as np
import scipy.stats
import sys
import time
import os
import scipy.optimize

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)	

from GW_generator import *
from GW_helper import * 	#routines for dealing with datasets

def compute_mismatch(phi_ref, h_true, theta, times, generator, modes):
	"Compute mismatch between h_true and h_mlgw as a function of phi_ref"
	theta = np.array(theta)
	theta[6] = phi_ref
	h_p, h_c = generator.get_WF(theta, times, modes)
	h_rec = h_p +1j* h_c
	F = compute_optimal_mismatch(h_rec,h_true, False)[0][0]
	return F

modes_to_k = lambda modes:[int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes] # [(l,m)] -> [k]

###############################

load = True
plot = False

N_waves = 1500
filename = "accuracy/mismatch_hist_TEOBResumS.dat"
filename_theta = "accuracy/theta_hist_TEOBResumS.dat"

f_min = 10

np.random.seed(23)

modes = [(2,2), (3,3), (3,2), (4,4)]

if not load:

	F = np.zeros((N_waves,len(modes)+1))  #[F_tot, [F_modes]]
	generator = GW_generator("TD_models/model_0")

		#getting random theta
	m1_range = (10.,100.)
	m2_range = (10.,100.)
	s1_range = (-0.8,0.8)
	s2_range = (-0.8,0.8)
	d_range = (.5,10.)
	i_range = (0,np.pi) 
	phi_0_range = (0,0)

	low_list = [m1_range[0],m2_range[0], s1_range[0], s2_range[0], d_range[0], i_range[0], phi_0_range[0]]
	high_list = [m1_range[1],m2_range[1], s1_range[1], s2_range[1], d_range[1], i_range[1], phi_0_range[1]]

	theta = np.random.uniform(low = low_list, high = high_list, size = (N_waves, 7))

	for i in range(N_waves):
		#times, h_p, h_c = generate_waveform(*theta[i,:], f_min = f_min, t_step = 1e-5, approx = "SEOBNRv2_opt")
		times, h_p_TEOB, h_c_TEOB, hlm, t_m = generate_waveform_TEOBResumS(*theta[i,:-1], f_min = f_min,
								verbose = False, t_step = 1e-4, modes = modes)


			#computing overall mismatch
		h_TEOB = h_p_TEOB + 1j*h_c_TEOB
		print("it: {} \t Theta: ".format(i), theta[i,:])
		res = scipy.optimize.minimize_scalar(compute_mismatch, bounds = [0.,2*np.pi],
						args = (h_TEOB, theta[i,:], times, generator, modes), method = "Brent")	
		F[i,0] = res['fun']
		print("\tOverall mismatch: ", F[i,0])

			#plotting
		if plot:
			temp_theta = np.concatenate((theta[i,:6],[res['x']]))
			h_p_mlgw, h_c_mlgw = generator.get_WF(temp_theta, times, modes)
			temp_theta = np.concatenate((np.round(temp_theta[:5],1), np.round(temp_theta[5:],2)))

			plt.figure()
			plt.title(r"$h_{+} \;\;\; \theta =$"+"{}".format(temp_theta))
			plt.plot(times, h_p_mlgw, label = "mlgw")
			plt.plot(times, h_p_TEOB, label = "TEOB")
			plt.legend(loc = 'upper left')

			plt.figure()
			plt.title(r"$h_{\times} \;\;\; \theta =$"+"{}".format(temp_theta))
			plt.plot(times, h_c_mlgw, label = "mlgw")
			plt.plot(times, h_c_TEOB, label = "TEOB")
			plt.legend(loc = 'upper left')
		

			#computing modes mismatch
		amp_aligned, ph_aligned = generator.get_modes(theta[i,:],times, out_type = "ampph", modes = modes, align22 = True)
		shifts = generator.get_shifts(theta[i,:], modes)

		for j in range(len(modes)):
			k = modes_to_k([modes[j]])

				#time aligning by hand... Really Ugly!!
			amp_lm = hlm[str(k[0])][0]
			extrema = scipy.signal.argrelextrema(np.abs(amp_lm), np.greater)
			t_lm = times - times[extrema[0][0]] #aligned grid

			amp, ph = generator.get_modes(theta[i,:], t_lm, out_type = "ampph", modes = modes[j], align22 = False)

			h = amp * np.exp(1j*ph)
			h_TEOB = hlm[str(k[0])][0] * np.exp(1j * hlm[str(k[0])][1])
			F_temp, phi_0 = compute_optimal_mismatch(h,h_TEOB) #things are aligned up to merger...
			print("\t",modes[j],F_temp)
			F[i,j+1] = F_temp

			#plotting
			if plot:			
				plt.figure()
				plt.title("Amp mode {}".format(str(modes[j])))
				plt.plot(t_lm, amp, label = "mlgw")
				plt.plot(t_lm, amp_aligned[:,j], label = "mlgw - aligned")	
				plt.plot(t_lm, hlm[str(k[0])][0], label = "TEOB")
				plt.legend(loc = 'upper left')
			
				plt.figure()
				plt.title("Ph mode {}".format(str(modes[j])))
				plt.plot(t_lm, ph, label = "mlgw")
				plt.plot(t_lm, ph_aligned[:,j], label = "mlgw - aligned")	
				plt.plot(t_lm, hlm[str(k[0])][1], label = "TEOB")
				plt.legend(loc = 'upper left')

		plt.show()


	f=open(filename,'ab') #to append
	if os.stat(filename).st_size == 0:
		np.savetxt(f,F, header = str(modes)) #putting header
	else:
		np.savetxt(f,F)
	f.close()

	f_theta=open(filename_theta,'ab') #to append
	if os.stat(filename_theta).st_size == 0:
		np.savetxt(f_theta, theta, header = str(modes)) #putting header
	else:
		np.savetxt(f_theta, theta)
	f_theta.close()


theta = np.loadtxt(filename_theta) #(N,7)
F = np.loadtxt(filename) #(N, len(modes)+1)
print("Loading histogram from: {}\n\t{} datapoints".format(filename, F.shape[0]))

plt.figure(figsize = (6.4, 4.8/2.))
plt.title("Overall mismatch", fontsize = 15)
plt.hist(np.log10(F[:,0]+1e-22), bins = 70, color = 'k', density = True)
plt.xlabel(r"$\log \; \bar{\mathcal{F}}$")
plt.savefig("accuracy/mismatch_overall.pdf", format = 'pdf')

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


import scipy.stats


fig, ax_list = plt.subplots(figsize = (6.4,6.4), nrows = 2, ncols = 2)
plt.subplots_adjust(hspace = .8, wspace = 0.7)
plt.suptitle("HMs contour plots", fontsize = 15)

for i in range(2):
	for j in range(2):
		k = 2*i +j
		ax_list[i,j].set_title("Mode {}".format(modes[k]), fontsize = 12)
		n_bins = 12
		q = np.maximum(theta[:,0]/theta[:,1],theta[:,1]/theta[:,0])
		mean, xedges, yedges, binnnumber = scipy.stats.binned_statistic_2d(q, theta[:,2], F[:,k+1], bins = (n_bins,n_bins))
		mean = mean.T #this is required for pcolormesh
		im = ax_list[i,j].pcolormesh(*np.meshgrid(xedges,yedges), np.log10(mean),  cmap='coolwarm')
		cb = plt.colorbar(im, ax = ax_list[i,j])
		ax_list[i,j].set_xlabel(r"$q$")
		ax_list[i,j].set_ylabel(r"$s_1$")

plt.savefig("accuracy/countor_HMs.pdf", format = 'pdf')

plt.show()














