import sys
sys.path.insert(0, '../../mlgw_v2')

import GW_generator as g
import lal
import numpy as np

model = g.GW_generator(0)
model.get_spherical_harmonics((2,2), 0,0)

for _ in range(1000):

	l = np.random.choice([2,3,4,5,6])
	m = np.random.choice([i for i in range(-l,l+1)])
	mode = (l,m)

	iota = np.arccos(np.random.uniform(-1,1))
	phi = np.random.uniform(0, np.pi*2)

	Y_lm_real, Y_lm_imag = model.get_spherical_harmonics(mode, iota, phi)
	Y_lm = Y_lm_real +1j* Y_lm_imag

	Y_lm_lal = lal.SpinWeightedSphericalHarmonic(iota, phi, -2, int(l), int(m))
	Y_lm_real_lal, Y_lm_imag_lal = Y_lm_lal.real, Y_lm_lal.imag

	print('###')
	print('mlgw\t',Y_lm_real, Y_lm_imag)
	print('lal\t', Y_lm_real_lal, Y_lm_imag_lal)
	print(np.abs(Y_lm)/np.abs(Y_lm_lal))
