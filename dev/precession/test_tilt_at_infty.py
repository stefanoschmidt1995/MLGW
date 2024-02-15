import lalsimulation as lalsim
import lalsimulation.tilts_at_infinity as tilt_infty
import lal
import matplotlib.pyplot as plt
import numpy as np

res_list = []

for i in range(100):
	
	q = 3 #np.random.uniform(1, 10)
	m1, m2 = q*20/(1+q), 20/(1+q)
	
	
	t1 = 2.#np.random.uniform(0, np.pi)
	t2, phi12 = 0., 0.
	s1, s2 = 0.8, 0.6 #np.random.uniform(0,1,2)
	fref = np.random.uniform(1,50)
	
	print('#########\nSpins ', s1, s2, fref)
	print('tilts start ', t1, t2)
	out_dict = tilt_infty.hybrid_spin_evolution.calc_tilts_at_infty_hybrid_evolve(m1*lalsim.lal.MSUN_SI, m2*lalsim.lal.MSUN_SI,
			s1,
			s2,
			t1, t2, phi12,
			fref,
			version ='v2')
	print('tilts end ', out_dict['tilt1_inf'], out_dict['tilt2_inf'])

	res_list.append((fref, out_dict['tilt2_inf']))

res_list = np.array(res_list)

plt.scatter(*res_list.T)
plt.show()
