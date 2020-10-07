from GW_generator import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)

gen = GW_generator("TD_models/model_0")

gen.list_modes()

theta = np.array([20,10,0.5,-0.3, 1, .34, 0]) #list of parameters to be given to generator [m1,m2,s1,s2]
times = np.linspace(-8,0.02, 100000) #time grid: peak of 22 mode at t=0
h_p, h_c = gen.get_WF(theta, times, modes = [(2,2),(3,3),(3,2)]) #returns amplitude and phase of the wave

amp_22, ph_22 = gen.get_modes(theta, times, modes = [(3,3)]) #(D,2)

	#plotting the wave
plt.figure(figsize=(15,8))
plt.title("GW by a BBH with [m1,m2,s1,s2] = "+str(theta[:4]), fontsize = 15)
plt.plot(times, h_p, c='k') #plot the plus polarization
plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel(r"$h_+$", fontsize = 12)
axins = inset_axes(plt.gca(), width="70%", height="30%", loc=2, borderpad = 2.)
axins.plot(times[times >= -0.2], h_p[times >= -0.2], c='k')

	#plotting 22 mode
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize =(15,8))
plt.suptitle("22 mode for a BBH with [m1,m2,s1,s2] = "+str(theta[:4]), fontsize = 15)
ax1.plot(times, amp_22, c='k')
ax1.set_ylabel(r"$A_{22}$", fontsize = 12)
ax2.plot(times, ph_22, c='k')
ax2.set_ylabel(r"$\phi_{22}$", fontsize = 12)
ax2.set_xlabel("Time (s)", fontsize = 12)
	#insets
axins1 = ax1.inset_axes([0.05, .6,.7,.3])
axins1.plot(times[times >= -0.2], amp_22[times >= -0.2], c='k')
axins2 = ax2.inset_axes([0.05, .6,.7,.3])
axins2.plot(times[times >= -0.2], ph_22[times >= -0.2], c='k')


plt.show()
