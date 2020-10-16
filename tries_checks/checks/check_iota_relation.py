import mlgw.GW_generator as gen
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../mlgw_v1') #folder in which every relevant routine is saved
import scipy.optimize

from GW_helper import *

q_range = (1.,20.)
m2_range = (5.,5.)
s1_range = (-0.8,0.95)
s2_range = (-0.8,0.95)

iota_range = np.linspace(0,np.pi,12)

N_WFs = 5

t_ms = np.zeros((N_WFs,len(iota_range)))

for i in range(N_WFs):
	q = np.random.uniform(*q_range)
	M = np.random.uniform(*m2_range)
	s1 = np.random.uniform(*s1_range)
	s2 = np.random.uniform(*s2_range)
	for j in range(len(iota_range)):
		times, h_p, h_c, t_m = generate_waveform_TEOBResumS(q*M/(1+q),M/(1+q),s1,s2, 1., iota_range[j], 0., t_min = 10., t_step = 1e-4)
		t_ms[i,j] = t_m

t_ms = (t_ms.T - t_ms[:,0]).T
plt.plot(iota_range, t_ms.T,'-', ms = 2)
plt.show()
