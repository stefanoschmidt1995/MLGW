import lalsimulation as lalsim
import lalsimulation.tilts_at_infinity as tilt_infty
import lal
import matplotlib.pyplot as plt
import numpy as np

chi = lambda q, S1, S2, t1, t2: (1+q)*S1*np.cos(t1)+(1+q**(-1))*S2*np.cos(t2)

res_list = []

for i in range(100):
	
	q = np.random.uniform(1, 10)
	m1, m2 = q*1/(1+q), 1/(1+q)
	
	
	t1 = np.random.uniform(0, np.pi)
	t2, phi12 = 0, 0 #np.pi/2., 0.
	s1, s2 = np.random.uniform(0.1,1,2)
	fref = np.random.uniform(1,50)
	
	print('#########\nSpins ', s1, s2, fref)
	print('tilts start ', t1, t2)
	
	S1 = s1*m1**2
	S2 = s2*m2**2
	q = 1/q

	chi_start = chi(q, S1, S2, t1,t2)
	k_inf = (np.cos(t1)*S1*(q**(-1)-q)+chi_start)/(1+q**(-1))

	out_dict = tilt_infty.hybrid_spin_evolution.calc_tilts_at_infty_hybrid_evolve(m1*lalsim.lal.MSUN_SI, m2*lalsim.lal.MSUN_SI,
			s1,
			s2,
			t1, t2, phi12,
			fref,
			version ='v2')
	chi_end = chi(q, S1, S2, out_dict['tilt1_inf'], out_dict['tilt1_inf'])
	t2_inf_pred = (chi_end - k_inf*(1+q))/(S2*q**(-1)-q)
	print('tilts end ', out_dict['tilt1_inf'], out_dict['tilt2_inf'])
	print('\t', t2_inf_pred)
	print(chi_start, chi_end)
	#res_list.append((fref, out_dict['tilt2_inf']))

quit()

res_list = np.array(res_list)

plt.scatter(*res_list.T)
plt.show()
