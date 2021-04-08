#Taken from https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/test___s_e_o_b_n_rv4_p_h_m__vs__4_h_m__ringdown_8py_source.html

import sys
import pytest
from scipy.interpolate import InterpolatedUnivariateSpline
import lalsimulation as ls
import lal
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../mlgw_v2')

import GW_generator as gen
from GW_helper import compute_optimal_mismatch

def get_SEOBNRv4HM_modes(q, M, chi1, chi2, f_start, deltaT):
	"""Generate SEOBNRv4HM modes"""
	m1SI = lal.MSUN_SI * q * M / (1.0 + q)
	m2SI = lal.MSUN_SI * M / (1.0 + q)
	nqcCoeffsInput = lal.CreateREAL8Vector(10)
	
	prefactor = 4.7864188273360336e-20 # G/c^2*(M_sun/Mpc)
	amp_prefactor = prefactor*M/1. # G/c^2 (M / d_L)
	nu = np.divide(q, np.square(1+q))
	distance = 1. * 1e6 * lal.PC_SI
	
	sphtseries, dyn, dynHI = ls.SimIMRSpinAlignedEOBModes(
		 deltaT,
		 m1SI,
		 m2SI,
		 f_start,
		 distance,
		 chi1,
		 chi2,
		 41,
		 0.0,
		 0.0,
		 0.0,
		 0.0,
		 0.0,
		 0.0,
		 0.0,
		 0.0,
		 1.0,
		 1.0,
		 nqcCoeffsInput,
		 0,
	 )
 
	# The minus sign in front of the modes takes into account the fact that the polarization basis in EOB
	# conventions is different wrt the one in LAL
	hI = {}
	modes = [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5)]
	for lm in modes:
		hI[lm] = np.trim_zeros(
		-1 * ls.SphHarmTimeSeriesGetMode(sphtseries, lm[0], lm[1]).data.data, "b"
	) /(amp_prefactor*nu)
 
	t = np.linspace(0, deltaT*len(hI[(2,2)]), len(hI[(2,2)]))
	t  = t - t[np.argmax(np.abs(hI[(2,2)]))]
 
	return t, hI
 
 
def get_SEOBNRv4PHM_modes(q, M, chi1, chi2, f_start, deltaT):
	"""Generate SEOBNRv4PHM modes"""
	m1SI = lal.MSUN_SI * q * M / (1.0 + q)
	m2SI = lal.MSUN_SI * M / (1.0 + q)
	approx = ls.SEOBNRv4PHM
	
	prefactor = 4.7864188273360336e-20 # G/c^2*(M_sun/Mpc)
	amp_prefactor = prefactor*M/1. # G/c^2 (M / d_L)
	nu = np.divide(q, np.square(1+q))
	distance = 1. * 1e6 * lal.PC_SI
	
	hlm = ls.SimInspiralChooseTDModes(0.,
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
		 approx
	)
	hI = {}
	modes = [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5)]
	for lm in modes:
		hI[lm] = ls.SphHarmTimeSeriesGetMode(hlm, lm[0], lm[1]).data.data /(amp_prefactor*nu)
 
	t = np.linspace(0, deltaT*len(hI[(2,2)]), len(hI[(2,2)]))
	t  = t - t[np.argmax(np.abs(hI[(2,2)]))]
	return t, hI
	
q = 1.6
M = 10.
chi1 = [0.,0.,.3]
chi2 = [0.,0.,.13]

#deltaT = 1./(100.*4096.)
#f_start = 500
deltaT = 1./(4.*4096.)
f_start = 40

t_P, hI_P = get_SEOBNRv4PHM_modes(q, M, chi1, chi2, f_start = f_start, deltaT = deltaT )
t_NP, hI_NP = get_SEOBNRv4HM_modes(q, M, chi1[2], chi2[2], f_start = f_start, deltaT = deltaT )

g = gen.GW_generator(1)

theta = np.array([M*q/(1+q),M/(1+q), chi1[0], chi1[1], chi1[2], chi2[0], chi2[1], chi2[2] ])
#h_p_mlgw_T, h_c_mlgw_T = g.get_twisted_modes(theta, t_P, modes = [(2, 2), (2, 1)], f_ref = f_start, alpha0 = -np.pi/2., gamma0 = np.pi/2.)

h_p_mlgw, h_c_mlgw = g.get_modes(theta[[0,1,4,7]], t_P, modes = [(2, 2), (2, 1)], out_type='realimag')

print("Mismatch SEOBNRv4PHM vs SEOBNRv4HM (22, 21)",compute_optimal_mismatch(hI_P[(2,2)], np.interp(t_P, t_NP, hI_NP[(2,2)]))[0][0], compute_optimal_mismatch(hI_P[(2,1)], np.interp(t_P, t_NP, hI_NP[(2,1)]))[0][0] )
print("Mismatch SEOBNRv4PHM vs mlgw (22, 21)",compute_optimal_mismatch(hI_P[(2,2)],h_p_mlgw[:,0]+1j*h_c_mlgw[:,0] )[0][0], compute_optimal_mismatch(hI_P[(2,1)], h_p_mlgw[:,1]+1j*h_c_mlgw[:,1])[0][0] )

plt.figure()
plt.title("(2,1) mode")
plt.plot(t_P, hI_P[(2,1)], label = 'P')
plt.plot(t_NP, hI_NP[(2,1)], label = 'NP')
plt.plot(t_P, h_p_mlgw[:,1], label = 'mlgw')
#plt.plot(t_P, h_p_mlgw_T[:,1], label = 'mlgw-T')
plt.legend()
plt.show()





 
