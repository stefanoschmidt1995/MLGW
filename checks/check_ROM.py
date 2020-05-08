import numpy as np
import lal
import lalsimulation as lalsim
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from GW_helper import * 	#routines for dealing with datasets

q = 1.
s1 = .0
m1 = 5.0
m1c = (m1*q*m1)**(3./5.)/(m1+m1*q)**(1./5.)
m2 = 10.0
m2c = (m2*q*m2)**(3./5.)/(m2+m2*q)**(1./5.)
m1tot = (1+q)*m1
m2tot = (1+q)*m2
f_min = 10.

#print(lalsim.GetApproximantFromString('TEOBResum_ROM'))

hptilde, hctilde = lalsim.SimInspiralChooseTDWaveform( #where is its definition and documentation????
		m1*lalsim.lal.MSUN_SI, #m1
		m2*lalsim.lal.MSUN_SI, #m2
		0., 0., s1, #spin vector 1
		0., 0., 0., #spin vector 2
		1.*1e6*lalsim.lal.PC_SI, #distance to source
		0., #inclination
		0., #phi ref
		0., #longAscNodes
		0., #eccentricity
		0., #meanPerAno
		5e-5, # time incremental step
		f_min, # lowest value of freq
		f_min, #some reference value of freq (??)
		lal.CreateDict(), #some lal dictionary
		lalsim.GetApproximantFromString('SEOBNRv2_opt') #approx method for the model
		)

h =  (hptilde.data.data+1j*hctilde.data.data)
(indices, ) = np.where(np.abs(h)!=0) #trimming zeros of amplitude
h = h[indices]

t = np.linspace(0.0, h.shape[0]*5e-5, h.shape[0])  #time actually
t_m =  t[np.argmax(np.abs(h))]
t = t - t_m #[-???,??]



import matplotlib.pyplot as plt

fig = plt.figure()
plt.title('rescaled times')
ax = fig.add_subplot(111)
ax.plot(t, h.real, color='k')

plt.show()






