import mlgw
import numpy as np
import matplotlib.pyplot as plt

	#generating the wave
gen = mlgw.GW_generator() #creating an istance of the generator (using default model)
theta = np.array([20,10,0.5,-0.3]) #list of parameters to be given to generator [m1,m2,s1,s2]
times = np.linspace(-8,0.02, 100000) #time grid: peak of 22 mode at t=0
modes = [(2,2), (3,3), (4,4), (5,5)]
h_p, h_c = gen.get_WF(theta, times, modes) #returns amplitude and phase of the wave

	#plotting the wave
fig = plt.figure(figsize=(15,8))
plt.title("GW by a BBH with [m1,m2,s1,s2] = "+str(theta), fontsize = 15)
plt.plot(times, h_p, c='k') #plot the plus polarization
plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel(r"$h_+$", fontsize = 12)
axins = fig.add_axes([0.2, 0.6, 0.4, 0.25])
axins.plot(times[times >= -0.2], h_p[times >= -0.2], c='k')
plt.show()
