import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../mlgw_v2')

import GW_generator as gen
from GW_helper import compute_optimal_mismatch

import lal
import lalsimulation as lasim

def get_SEOBNRv4PHM_modes(q, M, chi1, chi2, f_start , deltaT = 1./(4096*4.)):
	"""Generate SEOBNRv4PHM modes"""
	prefactor = 4.7864188273360336e-20 # G/c^2*(M_sun/Mpc)
	distance = 1. * 1e6 * lal.PC_SI  # 1 Mpc in m
	amp_prefactor = prefactor*M/1. # G/c^2 (M / d_L)
	nu = q/(1+q)**2
 
	m1SI = lal.MSUN_SI * q * M / (1.0 + q)
	m2SI = lal.MSUN_SI * M / (1.0 + q)
	approx = lasim.SEOBNRv4PHM
	hlm = lasim.SimInspiralChooseTDModes(0.,
		deltaT,
		m1SI,
		m2SI,
		chi1[0],
		chi1[1],
		chi1[2],
		chi2[0],
		chi2[1],
		chi2[2],
		f_start,
		f_start,
		distance,
		None,
		5,
		approx,
	)
	hI = {}
	modes = [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5)]
	for lm in modes:
		hI[lm] = lasim.SphHarmTimeSeriesGetMode(hlm, lm[0], lm[1]).data.data /amp_prefactor / nu
 
	times = np.linspace(0,len(hI[(2,2)])*deltaT,len(hI[(2,2)]))
	h_22 = hI[(2,2)]
	t_max = times[np.argmax(np.abs(h_22))]
	times = times-t_max
	return times, hI

compare_smooth = False
gen.list_models()
g = gen.GW_generator(1)

theta = np.array([20, 10,-0.2,.0,.1,.0,0.5,-0.2])#,[20,15,0.,.3,.1,.1,0.,-0.2]]
q = theta[0]/theta[1]
M = theta[0]+theta[1]
chi1 = theta[2:5]
chi2 = theta[5:]

	#computing modes SEOB
if compare_smooth:
	t_grid = np.linspace(-8,0.1,10000)
else:
	t_grid, hI_SEOB = get_SEOBNRv4PHM_modes(q, M, chi1, chi2, f_start = 20 , deltaT = 1./(4096))

	#computing modes mlgw
modes = [(2,1),(2,2)]

theta_NP = np.concatenate([theta[None,:2], np.linalg.norm(theta[None,2:5],axis = 1)[:,None], np.linalg.norm(theta[None,5:8],axis = 1)[:,None]] , axis = 1)[0,:]
h_p_mlgw, h_c_mlgw = g.get_twisted_modes(theta,t_grid, modes)
amp_NP_mlgw, ph_NP_mlgw = g.get_modes(theta_NP,t_grid, modes[0], out_type = 'ampph')

if compare_smooth:
	h_p_mlgw_smooth, h_c_mlgw_smooth = g.get_twisted_modes(theta,t_grid, modes, True) #smooth
	h = h_p_mlgw[:,0] +1j* h_c_mlgw[:,0]
	h_smooth = h_p_mlgw_smooth[:,0] +1j* h_c_mlgw_smooth[:,0]
	h_smooth = h_smooth*np.exp(1j*2.)
	F, ph = compute_optimal_mismatch(h,h_smooth)

	print("mismatch: ", F)
	
	plt.plot(t_grid, h.real, label = '2,1')
	plt.plot(t_grid, (h_smooth*np.exp(1j*ph)).real, label = '2,1 - smooth')
	plt.plot(t_grid, amp_NP_mlgw, label = '2,1 - NP')

else:
	plt.plot(t_grid, h_p_mlgw[:,0], label = '2,1')
	plt.plot(t_grid, hI_SEOB[(2,1)].real, label = '2,1 - SEOB')
	plt.plot(t_grid, amp_NP_mlgw, label = '2,1 - NP')


plt.legend()
plt.show()













