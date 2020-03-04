import mlgw.GW_generator as generator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)

#generating the wave
generator = generator.GW_generator() #creating an istance of the generator
theta = np.array([20,10,0.5,-0.3]) #list of parameters to be given to generator [m1,m2,s1,s2]
times = np.linspace(-8,0.02, 100000) #time grid at which waves shall be evaluated
h_p, h_c = generator.get_WF(theta, times) #returns amplitude and phase of the wave

generator.model_summary() #printing model summary

#plotting the wave
plt.figure(figsize=(15,8))
plt.title("GW by a BBH with [m1,m2,s1,s2] = "+str(theta), fontsize = 15)
plt.plot(times, h_p[0,:], c='k') #plot the plus polarization
plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel(r"$h_+$", fontsize = 12)
axins = inset_axes(plt.gca(), width="70%", height="30%", loc=2, borderpad = 2.)
axins.plot(times[times >= -0.2], h_p[0,times >= -0.2], c='k')
plt.show()
