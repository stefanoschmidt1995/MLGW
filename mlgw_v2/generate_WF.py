from GW_generator import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)

gen = GW_generator("TD_models/model_0")

gen.list_modes()

theta = np.array([20,10,0.5,-0.3]) #list of parameters to be given to generator [m1,m2,s1,s2]
times = np.linspace(-8,0.02, 100000) #time grid: peak of 22 mode at t=0
h_p, h_c = gen.get_WF(theta, times, modes = [(3,3), (2,2), (3,2)]) #returns amplitude and phase of the wave

#plotting the wave
plt.figure(figsize=(15,8))
plt.title("GW by a BBH with [m1,m2,s1,s2] = "+str(theta), fontsize = 15)
plt.plot(times, h_p, c='k') #plot the plus polarization
plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel(r"$h_+$", fontsize = 12)
axins = inset_axes(plt.gca(), width="70%", height="30%", loc=2, borderpad = 2.)
axins.plot(times[times >= -0.2], h_p[times >= -0.2], c='k')
plt.show()
