import mlgw
import mlgw.GW_generator as generator
import mlgw.EM_MoE as MoE
import numpy as np
import os
import matplotlib.pyplot as plt



generator = generator.GW_generator("TD")
help(generator)

theta = np.array([20.,10.,0.5,-0.3])
times = np.linspace(-10,0.016, 100000) #time grid at which waves shall be evaluated
amp, ph = generator.get_WF(theta, plus_cross = False, x_grid = times, red_grid = False)

plt.plot(times, amp[0,:]*np.exp(1j*ph[0,:]))
plt.show()
