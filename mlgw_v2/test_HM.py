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
	print("Using local version")
except:
	from mlgw.GW_generator import *	
	from mlgw.GW_helper import *
	print("Using pip package")

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


load = False		#whether to load the saved data
plot = True		#whether to plot the comparison between the WFs

N_waves = 1500 #WFs to generate
filename = "accuracy/mismatch_hist_TEOBResumS_new.dat"		#file to save the accuracy data to
filename_theta = "accuracy/theta_hist_TEOBResumS_new.dat"	#file to save the orbital paramters the hist refers to

f_min = 10	#starting frequency for the WFs

np.random.seed(2423) #optional, setting a random seed for reproducibility

modes = [(2,2), (2,1), (3,3), (3,2), (3,1), (4,4), (4,3), (4,2), (4,1), (5,5)]	#modes to inspect

if not load:

	F = np.zeros((N_waves,len(modes)+1))  #[F_tot, [F_modes]
	generator = GW_generator(0)	#initializing the generator with standard model

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
			#computing test WFs
			#add support also for lal WFs, 'cause you need to check!
		times, h_p_TEOB, h_c_TEOB, hlm, t_m = generate_waveform_TEOBResumS(*theta[i,:-1], f_min = f_min,
								verbose = False, t_step = 1e-4, modes = modes)
		h_TEOB = h_p_TEOB + 1j*h_c_TEOB

			#computing overall mismatch
		q = np.maximum(theta[i,0]/theta[i,1],theta[i,1]/theta[i,0])
		nu = np.divide(q, np.square(1+q))
		print("it: {} \n  Theta: {}\n   q: {}".format(i, theta[i,:],q))

		ids = np.where(times < 1e-4)[0] #cutting TEOB WFs beacause they have problems...
		res = scipy.optimize.minimize_scalar(compute_mismatch, bounds = [0.,2*np.pi],
						args = (h_TEOB[ids], theta[i,:], times[ids], generator, modes), method = "Brent")	
		F[i,0] = res['fun']
		print("\tOverall mismatch: ", F[i,0])

			#plotting if it is the case
		if plot:
			temp_theta = np.concatenate((theta[i,:6],[res['x']]))
			h_p_mlgw, h_c_mlgw = generator.get_WF(temp_theta, times, modes)
			temp_theta = np.concatenate((np.round(temp_theta[:5],1), np.round(temp_theta[5:],2)))

			fig, ax_list = plt.subplots(num=0, nrows = 2, ncols = 1, sharex = True)
			plt.suptitle(r"$\theta =$"+"{}".format(temp_theta))
			ax_list[0].set_title(r"$h_{+}$")
			ax_list[0].plot(times, h_p_mlgw, label = "mlgw")
			ax_list[0].plot(times, h_p_TEOB, label = "TEOB")
			ax_list[0].legend(loc = 'upper left')

			ax_list[1].set_title(r"$h_{\times}$")
			ax_list[1].plot(times, h_c_mlgw, label = "mlgw")
			ax_list[1].plot(times, h_c_TEOB, label = "TEOB")
			ax_list[1].legend(loc = 'upper left')

			#computing mismatch for each mode
		amp_mlgw, ph_mlgw = generator.get_modes(theta[i,:], times, 
					out_type = "ampph", modes = modes) #getting single modes from the GW_generator

		for j in range(len(modes)):
			k = modes_to_k([modes[j]])

			#removing the shit of TEOBResumS
			where_zero = np.where(amp_mlgw[:,j] == 0)[0]
			amp_TEOB =  hlm[str(k[0])][0]*nu #TEOBResumS convention is weird...
			ph_TEOB = hlm[str(k[0])][1]
			amp_TEOB[where_zero] = 0
			ph_TEOB[where_zero] = ph_TEOB[where_zero[0]]

				#mismatch with WFs aligned authomatically
			F_temp, phi_0 = compute_optimal_mismatch(amp_mlgw[:,j] * np.exp(1j*ph_mlgw[:,j]), amp_TEOB*np.exp(1j*ph_TEOB)) 

				#printing and saving results
			print("\t",modes[j], F_temp)
			F[i,j+1] = F_temp

			if plot:
				l, m = modes[j]
				lm = str(l)+str(m)

				plt.figure(int(lm+'0'))
				plt.title("Amp mode {}".format(str(modes[j])))
				plt.plot(times, amp_mlgw[:,j], label = "mlgw")	
				plt.plot(times, amp_TEOB, label = "TEOB")
				plt.legend(loc = 'upper left')

				fig, ax_list = plt.subplots(num=int(lm+'1'), nrows = 2, ncols = 1, sharex = True)
				ax_list[0].set_title("Ph mode {}".format(str(modes[j])))
				ax_list[0].plot(times, ph_mlgw[:,j], label = "mlgw")
				ax_list[0].plot(times, ph_TEOB, label = "TEOB")
				ax_list[0].legend(loc = 'upper left')

				ax_list[1].set_title("Ph difference mode {}".format(str(modes[j])))
				ax_list[1].plot(times, (ph_mlgw[:,j] - ph_TEOB)- (ph_mlgw[0,0] - ph_TEOB[0]), label = "mlgw - TEOB")

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
plt.subplots_adjust(hspace = .8)
plt.title("Overall mismatch", fontsize = 15)
plt.hist(np.log10(F[:,0]), bins = 70, color = 'k', density = True)
plt.xlabel(r"$\log \; \bar{\mathcal{F}}$")
plt.xlim([-6,0])
plt.tight_layout()
plt.savefig("accuracy/mismatch_overall.pdf", format = 'pdf', transparent=True)

	#optimal mismatch
fig, ax_list = plt.subplots(figsize = (6.4,1.5*6.4), nrows = len(modes), ncols = 1, sharex = True)
plt.subplots_adjust(hspace = .8)
plt.suptitle("HMs mismatch", fontsize = 15)
ax_list[-1].set_xlabel(r"$\log \; \bar{\mathcal{F}}$")

for i in range(len(modes)):
	ax_list[i].hist(np.log10(F[:,i+1]), bins = 70, color = 'k', density = True)
	ax_list[i].set_title("Mode {}".format(modes[i]), fontsize = 12)


plt.savefig("accuracy/mismatch_HMs.pdf", format = 'pdf', transparent=True)


	#optimal mismatch
fig, ax_list = plt.subplots(figsize = (6.4,1.*6.4), nrows = int(len(modes)/2+.5), ncols = 2, sharex = True)
plt.subplots_adjust(hspace = 0.6)
plt.suptitle("HMs mismatch", fontsize = 15)
ax_list[-1,0].set_xlabel(r"$\log \; \bar{\mathcal{F}}$")
ax_list[-1,1].set_xlabel(r"$\log \; \bar{\mathcal{F}}$")

for i in range(ax_list.shape[0]):
	for j in range(ax_list.shape[1]):
		k = ax_list.shape[1]*i + j
		if k >= len(modes):
			ax_list[i,j].cla()
			ax_list[i,j].axis('off')
			continue
		ax_list[i,j].hist(np.log10(F[:,k+1]), bins = 70, color = 'k', density = True)
		ax_list[i,j].set_title("Mode {}".format(modes[k]), fontsize = 12)


plt.savefig("accuracy/mismatch_HMs_twocols.pdf", format = 'pdf', transparent=True)



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

plt.savefig("accuracy/countor_HMs.pdf", format = 'pdf', transparent=True)

plt.show()














