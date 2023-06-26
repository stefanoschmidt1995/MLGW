###################
#	Simple routine to display basic usage of the model
###################

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid.inset_locator import (InsetPosition, mark_inset)
try:
	from GW_generator import GW_generator
except:
	from mlgw.GW_generator import GW_generator

###################
#	generating the WFs
try:
	gen = GW_generator("TD_models/model_0") #loading a custom model saved in a given folder
except:
	list_models()	#list the available prefitted models
	gen = GW_generator(0) #this uses the default prefitted model (labeled by 0)


gen.list_modes() #print to screen the modes available in the generator

modes_to_use = [(2,2),(3,3),(3,2), (4,4),(5,5)] #HMs to include in the WFs. If None, every mode available is included.

theta = np.array([20,10,0.5,-0.3, 1, .34, 0.88]) #list of parameters to be given to generator [m1,m2,s1,s2, d_L, iota, phi]
times = np.linspace(-8,0.01, 4096*8) #time grid in seconds: peak of 22 mode at t=0 -- shape = (D,)
h_p, h_c = gen.get_WF(theta, times, modes = modes_to_use) #returns plus and cross polarization of the wave -- shape = (D,)
amp_lm, ph_lm = gen.get_modes(theta, times, modes = modes_to_use) 	#returns amplitude and phase of the K modes -- shape = (D,K)
																	#each mode is time-aligned s.t. the peak of 22 happens at t=0

	#WFs can be also generated in parellel: many orbital paramters can be provided with a single call
	#The generator accepts also just masses and spins (in this case d_L = 1 Mpc and iota = phi = 0)
theta_many = np.array([[20,10,0.5,-0.3],[34,19,-0.2,0.6],[10,49,0.7,0.1]]) #shape = (N,4)
h_p_many, h_c_many = gen.get_WF(theta_many, times, modes = modes_to_use) #shape = (N,D)
amp_lm_many, ph_lm_many = gen.get_modes(theta_many, times, modes = modes_to_use) #shape = (N,D,K)

	#the call option is also available, with a sligthly different call signature:
h_p_call, h_c_call = gen(times, m1 = 20, m2 = 10, spin1_x = 0., spin1_y=0., spin1_z=0.5, spin2_x=0., spin2_y=0., spin2_z=-0.3, D_L=1., i = .34, phi_0 = 0.88, long_asc_nodes = 0., eccentricity=0., mean_per_ano=0.)	#(D,)
		#The dependence on the longitudinal ascension node, the eccentricity, the mean periastron anomaly and the orthogonal spin components is not currently implemented and it is mainted for compatibility with lal.

###################
#	plotting the waves

fig, ax = plt.subplots(1,1, figsize=(15,8))
plt.title("GW by a BBH with [m1,m2,s1,s2, d_L, iota, phi] = "+str(theta)+"\nModes: "+str(modes_to_use)+"\n", fontsize = 15)
ax.plot(times, h_p, c='k') #plot the plus polarization
plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel(r"$h_+$", fontsize = 12)
	#zoom on the merger
#axins = ax.inset_axes(width="70%", height="30%", loc=2, borderpad = 2.)
axins = ax.inset_axes([0.05, .65,.7,.3])
axins.plot(times[times >= -0.2], h_p[times >= -0.2], c='k')

	#plotting a mode
mode_to_plot = (3,3) 	#setting the mode to display
id_mode = modes_to_use.index(mode_to_plot)	#getting the index in amp_lm at which the mode is stored
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize =(15,8))
plt.suptitle(str(mode_to_plot)+" mode for a BBH with [m1,m2,s1,s2, d_L, iota, phi] = "+str(theta), fontsize = 15)
ax1.plot(times, amp_lm[:,id_mode], c='k')
ax1.set_ylabel(r"$A_{22}$", fontsize = 12)
ax2.plot(times, ph_lm[:,id_mode], c='k')
ax2.set_ylabel(r"$\phi_{22}$", fontsize = 12)
ax2.set_xlabel("Time (s)", fontsize = 12)
	#zoom on the merger
axins1 = ax1.inset_axes([0.05, .6,.7,.3])
axins1.plot(times[times >= -0.2], amp_lm[times >= -0.2, id_mode], c='k')
axins2 = ax2.inset_axes([0.05, .6,.7,.3])
axins2.plot(times[times >= -0.2], ph_lm[times >= -0.2, id_mode], c='k')


plt.show()
