###################
#	Routine to test the WF generator accuracy, by computing the mismatch with train model TEOBResumS.
#	It saves the collected data on the accuracy and it plots fancy histograms
###################

import matplotlib.pyplot as plt 
import numpy as np
import sys
import os
import scipy.optimize
import scipy.stats

try:
	from GW_generator import *	#GW_generator: builds the WF
	from GW_helper import * 	#routines for dealing with datasets
except:
	from mlgw.GW_generator import *	
	from mlgw.GW_helper import * 	

def compute_mismatch(phi_ref, h_true, theta, times, generator, modes):
	"Compute mismatch between h_true and h_mlgw as a function of phi_ref"
	theta = np.array(theta)
	theta[6] = phi_ref
	h_p, h_c = generator.get_WF(theta, times, modes)
	h_rec = h_p +1j* h_c
	F = compute_optimal_mismatch(h_rec,h_true, False)[0][0]
	return F

modes_to_k = lambda modes:[int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes] # [(l,m)] -> [k]

###################
#	computing mismatch

load = True	#whether to load the saved data
plot = False	#whether to plot the comparison between the WFs

N_waves = 1500 #WFs to generate
filename = "accuracy/mismatch_hist_TEOBResumS.dat"		#file to save the accuracy data to
filename_theta = "accuracy/theta_hist_TEOBResumS.dat"	#file to save the orbital paramters the hist refers to

f_min = 10	#starting frequency for the WFs

np.random.seed(24) #optional, setting a random seed for reproducibility

modes = [(2,2), (2,1), (3,3), (3,2), (3,1), (4,4), (4,3), (4,2), (4,1), (5,5)]	#modes to inspect

if not load:

	F = np.zeros((N_waves,2*len(modes)+1))  #[F_tot, [F_modes], [F_modes_auto]]
	generator = GW_generator("TD_models/model_0")	#initializing the generator

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
		if np.maximum(theta[i,0]/theta[i,1], theta[i,1]/theta[i,0]) < 2:
			plot = True
		else:
			plot = False
			#computing test WFs
		times, h_p_TEOB, h_c_TEOB, hlm, t_m = generate_waveform_TEOBResumS(*theta[i,:-1], f_min = f_min,
								verbose = False, t_step = 1e-4, modes = modes)
		h_TEOB = h_p_TEOB + 1j*h_c_TEOB

			#computing overall mismatch
		print("it: {} \t Theta: ".format(i), theta[i,:])
		res = scipy.optimize.minimize_scalar(compute_mismatch, bounds = [0.,2*np.pi],
						args = (h_TEOB, theta[i,:], times, generator, modes), method = "Brent")	
		F[i,0] = res['fun']
		print("\tOverall mismatch: ", F[i,0])

			#plotting if it is the case
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

			#computing mismatch for each mode
		amp_aligned, ph_aligned = generator.get_modes(theta[i,:], times, 
					out_type = "ampph", modes = modes, align22 = True) #getting single modes from the GW_generator
		shifts = generator.get_shifts(theta[i,:], modes) #getting shifts

		for j in range(len(modes)):
			k = modes_to_k([modes[j]])

				#time aligning by hand... Really Ugly!!
			argpeak = locate_peak(hlm[str(k[0])][0])
			t_lm = times - times[argpeak] #aligned grid

			amp, ph = generator.get_modes(theta[i,:], t_lm, out_type = "ampph", modes = modes[j], align22 = False)

			h = amp * np.exp(1j*ph)
			h_TEOB = hlm[str(k[0])][0] * np.exp(1j * hlm[str(k[0])][1])
				#mismatch with WFs aligned by hand
			F_temp, phi_0 = compute_optimal_mismatch(h,h_TEOB)
				#mismatch with WFs aligned authomatically
			F_aligned, phi_0 = compute_optimal_mismatch(amp_aligned[:,j] * np.exp(1j*ph_aligned[:,j]), h_TEOB) 

				#printing and saving results
			print("\t",modes[j],F_temp, F_aligned)
			print("\t shift rec vs true: {} / {} ".format(shifts[j]*(theta[i,0]+theta[i,1]),times[argpeak]))
			F[i,j+1] = F_temp
			F[i,j+1+len(modes)] = F_aligned

				#plotting if it is the case
			if plot:
				plt.figure()
				plt.title("Amp mode {}".format(str(modes[j])))
				plt.plot(times, amp, label = "mlgw")
				plt.plot(times, amp_aligned[:,j], label = "mlgw - aligned")	
				plt.plot(times, hlm[str(k[0])][0], label = "TEOB")
				plt.legend(loc = 'upper left')
			
				plt.figure()
				plt.title("Ph mode {}".format(str(modes[j])))
				plt.plot(times, ph, label = "mlgw")
				plt.plot(times, ph_aligned[:,j], label = "mlgw - aligned")	
				plt.plot(times, hlm[str(k[0])][1], label = "TEOB")
				plt.legend(loc = 'upper left')

				plt.figure()
				plt.title("Ph difference mode {}".format(str(modes[j])))
				plt.plot(times, ph - hlm[str(k[0])][1], label = "mlgw - TEOB")
				plt.legend(loc = 'upper left')

		plt.show()

		#saving results to file (appending if necessary)
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

###################
#	plotting results

	#loading from file
theta = np.loadtxt(filename_theta) #(N,7)
F = np.loadtxt(filename) #(N, 2*len(modes)+1)

	#removing eventual zeros from the histograms
zeros = np.where(F[:,0]!=0)[0]
F = F[zeros,:]
theta = theta[zeros,:]

print("Loading histogram from: {}\n\t{} datapoints".format(filename, F.shape[0]))

#Plotting

	#overall mismatch
plt.figure(figsize = (6.4, 4.8/2.))
plt.title("Overall mismatch", fontsize = 15)
plt.hist(np.log10(F[:,0]), bins = 70, color = 'k', density = True)
plt.xlabel(r"$\log \; \bar{\mathcal{F}}$")
plt.savefig("accuracy/mismatch_overall.pdf", format = 'pdf')
plt.xlim([-6,0])

	#optimal mismatch
fig, ax_list = plt.subplots(figsize = (6.4,1.5*6.4), nrows = len(modes), ncols = 1, sharex = True)
plt.subplots_adjust(hspace = .8)
plt.suptitle("HMs mismatch", fontsize = 15)
ax_list[-1].set_xlabel(r"$\log \; \bar{\mathcal{F}}$")

for i in range(len(modes)):
	ax_list[i].hist(np.log10(F[:,i+1]), bins = 70, color = 'k', density = True)
	ax_list[i].set_title("Mode {}".format(modes[i]), fontsize = 12)


plt.savefig("accuracy/mismatch_HMs.pdf", format = 'pdf')

	#auto-aligned mismatch
fig, ax_list = plt.subplots(figsize = (6.4,1.5*6.4),nrows = len(modes), ncols = 1, sharex = True)
plt.subplots_adjust(hspace = .8)
plt.suptitle("HMs mismatch - auto aligned", fontsize = 15)
ax_list[-1].set_xlabel(r"$\log \; \bar{\mathcal{F}}$")

for i in range(len(modes)):
	ax_list[i].hist(np.log10(F[:,i+1+len(modes)]), bins = 70, color = 'k', density = True)
	ax_list[i].set_title("Mode {}".format(modes[i]), fontsize = 12)

plt.savefig("accuracy/mismatch_HMs_auto.pdf", format = 'pdf')

	#contour plots
fig, ax_list = plt.subplots(figsize = (1.5*6.4,1.5*6.4), nrows = 4, ncols = 3)
plt.subplots_adjust(hspace = .8, wspace = 0.7)
plt.suptitle("HMs contour plots", fontsize = 15)

for i in range(ax_list.shape[0]):
	for j in range(ax_list.shape[1]):
		k = ax_list.shape[1]*i +j
		if k >= len(modes):
			ax_list[i,j].cla()
			ax_list[i,j].axis('off')
			continue
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














