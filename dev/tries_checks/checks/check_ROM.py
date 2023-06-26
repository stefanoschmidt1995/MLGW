import numpy as np
import lal
import lalsimulation as lalsim
import sys
sys.path.insert(1, '../mlgw_v1') #folder in which every relevant routine is saved

from GW_helper import * 	#routines for dealing with datasets

import mlgw.GW_generator as gen

#remember to do:
#export LAL_DATA_PATH=/home/stefano/Documents/Stefano/scuola/uni/tesi_magistrale/code/checks/cbcrom-master/data
# all the relevant files are in the repository: https://git.ligo.org/lscsoft/lalsuite-extra/-/tree/master/data
# you need to download them and put in the right path...


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

phiRef=0.0
deltaT=1./2**14
fLow=50.0
fRef=50.0
distance=1.0e6*lal.PC_SI
inclination=0.0
m1SI=100.35*lal.MSUN_SI
m2SI=100.35*lal.MSUN_SI
lambda1=2000.0
lambda2=000.0

#help(lal)
#quit()
hp1, hc1 = lalsim.SimInspiralTEOBResumROM(phiRef,deltaT,fLow,fRef,distance,inclination,m1SI,m2SI,lambda1,lambda2)
#hp1, hc1 = lalsim.SimIMREOBNRv2HMROM(phiRef,deltaT,fLow,fRef,distance,inclination,m1SI,m2SI,0.,0.)

h =  (hp1.data.data+1j*hc1.data.data)
(indices, ) = np.where(np.abs(h)!=0) #trimming zeros of amplitude
h = h[indices]

t = np.linspace(0.0, h.shape[0]*deltaT, h.shape[0])  #time actually
t_m =  t[np.argmax(np.abs(h))]
t = t - t_m #[-???,??]

a = gen.GW_generator()
hp2, hc2 = a.get_WF([1.35,1.35,0,0],t)

import matplotlib.pyplot as plt

fig = plt.figure()
plt.title('rescaled times')
ax = fig.add_subplot(111)
ax.plot(t, h.real, color='k')
ax.plot(t, hp2[0,:], color='r')

plt.show()






