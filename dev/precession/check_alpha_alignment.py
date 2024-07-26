import mlgw
from mlgw import precession_helper
import numpy as np
import matplotlib.pyplot as plt


m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, inclination, phiref = 50, 7, 0.6*np.cos(1.), 0.6*np.sin(1.), 0.1, 4e-8, 0., 0.3, 0., 0.
M = 20

alpha_zeros = []
q_points = np.linspace(2, 10, 100)
#for q in np.random.uniform(2, 10, 10):

for q in q_points:
	#alpha, beta, gamma, times = precession_helper.get_IMRPhenomTPHM_angles(q*M/(1+q), M/(1+q), s1x, s1y, s1z, s2x, s2y, s2z,
	#	t_grid = None, fref = 40., fstart = 10)
	alpha, beta, gamma = precession_helper.get_IMRPhenomTPHM_angles(q*M/(1+q), M/(1+q), s1x, s1y, s1z, s2x, s2y, s2z,
		t_grid = np.linspace(-10, 0, 10000), fref = 10, fstart = 10)
	
	alpha_zeros.append(alpha[0])
	

#alpha2, beta2, gamma2, times2 = precession_helper.get_IMRPhenomTPHM_angles(q*M/(1+q), M/(1+q), s1x, s1y, s1z, s2x, s2y, s2z, t_grid = None, fref = 40., fstart = 10)
#plt.plot(times2, alpha2)
	#plt.plot(times, alpha, label = 'q = {}'.format(q))

plt.plot(q_points, alpha_zeros)

plt.legend()
plt.show()

