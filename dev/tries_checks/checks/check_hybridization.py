import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats
import sys
import time
sys.path.insert(1, '../mlgw_v1')

from MLGW_generator import *
from GW_helper import * 	#routines for dealing with datasets

N_waves = 5
	#true waves
theta, true_amp, true_ph, times = create_dataset_TD(N_waves, N_grid = 10000,
                t_coal = .5, q_range = (1.,5.), m2_range = (10.,11.), s1_range = (-0.8,0.8), s2_range = (-0.,0.),
                t_step = 5e-5, lal_approximant = "SEOBNRv2_opt")

generator = MLGW_generator("TD", "../mlgw_v1/models_TD_short_al_merger")

gen_amp, gen_ph = generator.get_WF(theta, plus_cross = False, x_grid = times, red_grid = True)

	#time to do hybridization
#getting indices of part to hybridise
m_c = (theta[0,0]* theta[0,1])**(3./5.)/(theta[0,0]+ theta[0,1])**(1./5.)
m_tot = theta[0,0]+ theta[0,1]
indices = np.where(gen_ph <= gen_ph[0,0]+1)[1]
print(indices.shape)

#phi_prime = (gen_ph[0,indices[-1]]- gen_ph[0,indices[-1]+1])/ ((times[indices[-1]]- times[indices[-1]+1]))
#alpha = phi_prime*8./5. * np.power(np.abs(times[indices[-1]])*m_tot,3./8.) /m_tot

#gen_ph[0,indices] = -alpha*  np.power(np.abs(times[indices]-times[indices[-1]])*m_tot, 5./8.) + gen_ph[0,indices[-1]+1]

#gen_ph[0,indices] = -4152*  (np.power(np.abs(times[indices])*m_tot/m_c, 5./8.) - np.power(np.abs(times[indices[-1]])*m_tot/m_c, 5./8.) ) + gen_ph[0,indices[-1]+1]

	#plotting
plt.figure()
#plt.plot(times*m_tot, gen_ph[0,:], label = "rec")
#plt.plot(times[indices]*m_tot, (times[indices]-times[indices[-1]])*m_tot*phi_prime+ gen_ph[0,indices[-1]] , label = "indices")
m=np.zeros(N_waves,)
for i in range(N_waves):
	m_tot = theta[i,0]+ theta[i,1]

	p = np.polyfit(np.log(-times[indices]*m_tot),np.log(-true_ph[i,indices]), 1)
	m[i] = p[0]
	print(i, p)
	if i ==0:
		#plt.plot(-times[indices]*m_tot,true_ph[i,indices], 'o', ms = 1,label = "true")
		plt.plot(times[indices]*m_tot,(-true_ph[i,indices]/np.power(-true_ph[i,indices], -1./4.)), 'o', ms = 1,label = "scaled")

#plt.plot(-times[indices], 30000*(-times[indices]/30)**(5./8.))
#plt.xscale("log")
#plt.yscale("log")
#plt.plot(times,gen_amp[0,:]*np.exp(1j*gen_ph[0,:]), label = "rec")
#plt.plot(times,true_amp[0,:]*np.exp(1j*true_ph[0,:]), label = "true")
plt.legend()

plt.figure()
plt.plot(theta[:,0]/theta[:,1], m, 'o')

plt.figure()
plt.plot(theta[:,2], m, 'o')

plt.figure()
plt.plot(theta[:,3], m, 'o')


plt.show()


