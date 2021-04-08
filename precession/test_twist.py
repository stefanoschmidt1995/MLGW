#The problem is that in SEOBNRv4PHM (and also TEOBRESUMS) beta is the angle between L(t) and L(t=0)
#I compute it as the angle between L(t) and J. This causes an inconsistency between the two models

#For a way to compute the PN eqs with lal (in the frame that you like the most), see: https://git.ligo.org/waveforms/reviews/lalsimulation.siminspiralspintaylorpnevolveorbit/-/blob/master/test_PN_vs_NR/test_vs_SXS.py


#HUGE PROBLEM:
#f_start in SimInspiralChooseTDModes is not the orbital frequnecy as computed in spin_evolution
#I do not know what it is... But it would be interesting to know that!!!!
#If you put the computed orbital frequency, i.e. omega_22 / 2 then it is all perfect and the two WFs matches

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../mlgw_v2')

import GW_generator as gen
from GW_helper import compute_optimal_mismatch, locate_peak

import lal
import lalsimulation as lalsim

import scipy.optimize, scipy.integrate, scipy.interpolate

def compute_mismatch(gamma0, h_true, theta, times, generator, modes, f_ref):
	"Compute mismatch between h_true and h_mlgw as a function of phi_ref"
	theta = np.array(theta)
	h_p, h_c = generator.get_twisted_modes(theta, times, modes, f_ref, gamma0)
	h_rec = h_p +1j* h_c
	F = compute_optimal_mismatch(np.squeeze(h_rec),np.squeeze(h_true), True)[0][0]
	print(F)
	return F

def get_TEOBResumS_WF(q, M, chi1, chi2, f_start , deltaT = 1./(4096*4.)):
	path_TEOBResumS = '/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/precession/TEOBResumS/Python'
	sys.path.insert(1,path_TEOBResumS) #path to local installation of TEOBResumS
	import EOBRun_module
	
	pars = {'M'                  : M,
			'q'                  : q,
			'Lambda1'            : 0.,
			'Lambda2'            : 0.,     
			'chi1x'               : 0.2,
			'chi1y'               : -0.5,
			'chi1z'               : 0.5,
			'chi2x'               : -0.2,
			'chi2y'               : 0.75,
			'chi2z'               : 0.5,
			'domain'             : 0,      # TD
			'arg_out'            : 1,     # Output hlm/hflm. Default = 0
			'use_mode_lm'        : [0,1],      # List of modes to use/output through EOBRunPy
			'srate_interp'       : 1./deltaT,  # srate at which to interpolate. Default = 4096.
			'use_geometric_units': 0,      # Output quantities in geometric units. Default = 1
			'initial_frequency'  : f_start,   # in Hz if use_geometric_units = 0, else in geometric units
			'interp_uniform_grid': 2,      # Interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
			'distance': 1.,
			'inclination': 0.,
		}
	time_full, h_p, h_c, hlm = EOBRun_module.EOBRunPy(pars)
	h_TEOB_22 = hlm['1'][0]*np.exp(1j*hlm['1'][1])
	h_TEOB = hlm['0'][0]*np.exp(1j*hlm['0'][1])

	argpeak = locate_peak(np.squeeze(np.abs(h_TEOB_22))) #aligned at the peak of the 22
	t_peak = time_full[argpeak]
	time_full = (time_full - t_peak)
	
	return time_full, h_TEOB, h_TEOB_22

def get_SEOBNRv4HM_WF(q, M, chi1, chi2, f_start , t_step = 1./(4096*4.)):
	m1 = q*M/(1+q)
	m2 = M/(1+q)
	prefactor = 4.7864188273360336e-20 # G/c^2*(M_sun/Mpc)
	amp_prefactor = prefactor*(M)/1. # G/c^2 (M / d_L)
	nu = np.divide(q, np.square(1+q))
	
	nqcCoeffsInput=lal.CreateREAL8Vector(10)
	sp, dyn, dynHi = lalsim.SimIMRSpinAlignedEOBModes(t_step, m1*lal.MSUN_SI, m2*lal.MSUN_SI, f_start, 1e6*lal.PC_SI, chi1, chi2,41, 0., 0., 0.,0.,0.,0.,0.,0.,1.,1.,nqcCoeffsInput, 0)
	print("WF generated")

	h_SEOB = {}
	while sp is not None:
		if (sp.l, sp.m) != (2,1) and (sp.l, sp.m) != (2,2):
			sp = sp.next
			continue
		hlm = sp.mode.data.data #complex mode vector
		hlm = hlm/ amp_prefactor / nu #scaling with the convention of SEOB
		h_SEOB[(sp.l, sp.m)] = hlm

		plt.plot(hlm)
		plt.show()


		if (sp.l, sp.m) == (2,2): #get grid
			amp_22 = np.abs(sp.mode.data.data) #amp of 22 mode (for time grid alignment)
			time_full = np.linspace(0.0, sp.mode.data.length*t_step, sp.mode.data.length) #time grid at which wave is computed
			argpeak = locate_peak(amp_22) #aligned at the peak of the 22
			t_peak = time_full[argpeak]
			time_full = (time_full - t_peak)
		sp = sp.next
	return time_full, h_SEOB

def get_SEOBNRv2_WF(q, M, chi1, chi2, f_start , deltaT = 1./(4096*4.)):
	m1 = q*M/(1+q)
	m2 = M/(1+q)
	hp, hc = lalsim.SimInspiralChooseTDWaveform( #where is its definition and documentation????
				m1*lalsim.lal.MSUN_SI, #m1
				m2*lalsim.lal.MSUN_SI, #m2
				0., 0., chi1, #spin vector 1
				0., 0., chi2, #spin vector 2
				1e6*lalsim.lal.PC_SI, #distance to source
				0., #inclination
				0., #phi
				0., #longAscNodes
				0., #eccentricity
				0., #meanPerAno
				deltaT, # time incremental step
				f_start, # lowest value of freq
				f_start, #some reference value of freq (??)
				lal.CreateDict(), #some lal dictionary
				lalsim.SimInspiralGetApproximantFromString('SpinTaylorT4') #approx method for the model
			)
	h_p, h_c =  hp.data.data, hc.data.data
	times = np.linspace(0,len(h_c)*deltaT,len(h_c))
	ph = np.unwrap(np.angle(h_p+1j*h_c))
	omega_22 = (ph[2]-ph[0])/(2*deltaT)
	print("FREQUENCY of THE WF (NP): ",omega_22, omega_22/(2*np.pi))
	t_max = times[np.argmax(np.abs(h_p+1j* h_c))]
	times = times-t_max
	return times, h_p+1j* h_c

def get_SEOBNRv4PHM_modes(q, M, chi1, chi2, f_start , deltaT = 1./(4096*4.)):
	#See the paper: https://arxiv.org/pdf/2004.09442.pdf
	"""Generate SEOBNRv4PHM modes"""
	prefactor = 4.7864188273360336e-20 # G/c^2*(M_sun/Mpc)
	distance = 1. * 1e6 * lal.PC_SI  # 1 Mpc in m
	amp_prefactor = prefactor*M/1. # G/c^2 (M / d_L)
	nu = q/(1+q)**2
 
	m1SI = lal.MSUN_SI * q * M / (1.0 + q)
	m2SI = lal.MSUN_SI * M / (1.0 + q)
	#approx = lalsim.SEOBNRv4PHM
	approx = lalsim.SimInspiralGetApproximantFromString('SEOBNRv4PHM')
	hlm = lalsim.SimInspiralChooseTDModes(0.,
		deltaT,
		m1SI,
		m2SI,
		chi1[0],
		chi1[1],
		chi1[2],
		chi2[0],
		chi2[1],
		chi2[2],
		f_start, #/**< starting GW frequency (Hz) */ #what is this f_start?? 
		f_start, #/**< reference GW frequency (Hz) */
		distance,
		None,
		5,
		approx,
	)
	hI = {}
	modes = [(2, 2), (2, 1), (2,-1), (3, 3), (4, 4), (5, 5)]
	for lm in modes:
		hI[lm] = lalsim.SphHarmTimeSeriesGetMode(hlm, lm[0], lm[1]).data.data /amp_prefactor / nu
 
	times = np.linspace(0,len(hI[(2,2)])*deltaT,len(hI[(2,2)]))
	h_22 = hI[(2,2)]
	ph = np.unwrap(np.angle(np.conj(h_22)))
	omega_22 = (ph[2]-ph[0])/(2*deltaT)
	print("FREQUENCY of THE WF: ",omega_22, omega_22/(2*np.pi))
	t_max = times[np.argmax(np.abs(h_22))]
	times = times-t_max
	return times, hI

gen.list_models()
g = gen.GW_generator(1)

#theta = np.array([.68, .32,0.1,.2,-0.3,-0.8,0.,0.32])#,[20,15,0.,.3,.1,.1,0.,-0.2]]
theta = np.array([.68, .32,0.,0.,0.3,-0.,0.,0.82])#,[20,15,0.,.3,.1,.1,0.,-0.2]]
#theta = np.array([2*6.8, 2*3.2,0.,.0,-0.3,-0.,0.,0.42])
q = theta[0]/theta[1]
M = theta[0]+theta[1]
chi1 = theta[2:5]
chi2 = theta[5:]

	#computing modes SEOB
#t_grid, h_SEOB = get_SEOBNRv2_WF(q, M, np.linalg.norm(chi1), np.linalg.norm(chi2), f_start = 400 , deltaT = 1./(4096*5))

if True:
	t_grid, hI_SEOB = get_SEOBNRv4PHM_modes(q, M, chi1, chi2, f_start = 400 , deltaT = 1./(409600))
	h_SEOB = hI_SEOB[(2,1)]
	h_SEOB_22 = hI_SEOB[(2,2)]
	np.savetxt('WF.dat', np.column_stack([t_grid, h_SEOB, h_SEOB_22]))
else:
	WFs = np.loadtxt('WF.dat', dtype = np.complex64)
	t_grid, h_SEOB, h_SEOB_22 = WFs[:,0], WFs[:,1], WFs[:,2]
	t_grid = t_grid.real

	#looking at SEOBNRv4HM
	#SEOBNRv4 is not consistent trhoughout its versions!!! Fuck it!!
	#SEOBNRv4HM and SEOBNRv4PHM do not agree on the NP limit!! Super weird
#t_grid, h_SEOB_NPHM = get_SEOBNRv4HM_WF(q, M, np.linalg.norm(chi1), np.linalg.norm(chi2), f_start = 40 , t_step = 1./(4.*4096))
#h_SEOB = h_SEOB_NPHM[(2,1)]
#h_SEOB_22 =  h_SEOB_NPHM[(2,2)]

#t_grid, h_SEOB, h_SEOB_22 = get_TEOBResumS_WF(q, M, chi1, chi2, f_start = 400 , deltaT = 1./(4096*5))

	#computing modes mlgw
modes = [(2,1),(2,2), (2,-1)]

theta_NP = np.concatenate([theta[None,:2], np.linalg.norm(theta[None,2:5],axis = 1)[:,None], np.linalg.norm(theta[None,5:8],axis = 1)[:,None]] , axis = 1)[0,:]

#res = scipy.optimize.minimize_scalar(compute_mismatch, bounds = [0.,2*np.pi],
#						args = (h_SEOB, theta, t_grid, g, modes[0], 400.), method = "Brent")	
#print("Optimal mismatch, gamma0: ",res['fun'], res['x'])

#h_p_mlgw_bis, h_c_mlgw_bis = g.get_twisted_modes(theta,t_grid, modes, 400.)
#h_p_mlgw, h_c_mlgw = g.get_modes(theta[[0,1,4,7]],t_grid, modes, out_type = 'realimag')
#h_mlgw_bis = h_p_mlgw_bis[:,0]+1j*h_c_mlgw_bis[:,0]
#F, ph = compute_optimal_mismatch(h_mlgw,h_mlgw_bis)
#print("Mismatch, ph ",F,ph)

h_p_mlgw, h_c_mlgw = g.get_twisted_modes(theta,t_grid, modes, 400., -np.pi/2., np.pi/2.)#,res['x'])
amp_NP_mlgw, ph_NP_mlgw = g.get_modes(theta_NP,t_grid, modes[0], out_type = 'ampph')

h_mlgw = h_p_mlgw[:,0]+1j*h_c_mlgw[:,0] #(2,1)
h_mlgw_22 = h_p_mlgw[:,1]+1j*h_c_mlgw[:,1] #(2,2)

F, ph = compute_optimal_mismatch(h_mlgw,h_SEOB)

#h_mlgw=h_mlgw*np.exp(-1j*1.5)

ph_mlgw = np.unwrap(np.angle(h_mlgw))
ph_SEOB = np.unwrap(np.angle(h_SEOB))
ph_diff = (ph_mlgw - ph_SEOB)
#ph_diff = (ph_mlgw-ph_mlgw[0] - (ph_SEOB-ph_SEOB[0]))

ph_mlgw_22 = np.unwrap(np.angle(h_mlgw_22))
ph_SEOB_22 = np.unwrap(np.angle(h_SEOB_22))
ph_diff_22 = (ph_mlgw_22-ph_mlgw_22[0] - (ph_SEOB_22-ph_SEOB_22[0]))

#h_mlgw = h_mlgw*np.exp(1j*alpha[0,:])

print('mismatch ',F, ph)

import lalintegrate_PNeqs
#is beta computed correctly???
alpha, beta = lalintegrate_PNeqs.get_alpha_beta(theta[0]/theta[1], theta[2:5],theta[5:8], 400., t_grid, f_merger = g.get_merger_frequency(theta_NP[None,:]))

	#gamma computation
alpha_dot = np.gradient(alpha, t_grid, axis = 1) #(N,D)
f = scipy.interpolate.interp1d(t_grid, np.multiply(alpha_dot, np.cos(beta)))
f_ = lambda t, y : -f(t)
res = scipy.integrate.solve_ivp(f_, (t_grid[0],t_grid[-1]), [0.], t_eval = t_grid)
gamma_ivp = res['y']

	#plotting everything
plt.plot(t_grid, (h_mlgw).real, label = '2,1')
plt.plot(t_grid, h_SEOB.real, label = '2,1 - SEOB')
plt.plot(t_grid, amp_NP_mlgw, label = '2,1 - NP')
plt.legend()

plt.figure()
plt.title("phase")
plt.plot(t_grid, ph_SEOB, label ='SEOB')
plt.plot(t_grid, ph_mlgw, label ='mlgw')
plt.legend()

plt.figure()
plt.title(r"$\Delta \Phi - \alpha, \beta, \gamma$")
plt.plot(t_grid, ph_diff, label = 'ph diff')
plt.plot(t_grid, alpha[0,:], label = 'alpha')
plt.plot(t_grid, beta[0,:], label = 'beta')
plt.plot(t_grid, gamma_ivp[0,:], label = 'gamma ivp')
#plt.plot(t_grid, np.cos(beta)[0,:], label = 'cos(beta)')
plt.legend()

#plt.plot(t_grid, ph_SEOB-ph_SEOB[0], label = '2,1 - SEOB')
#plt.plot(t_grid, (ph_mlgw-ph_mlgw[0] - (ph_SEOB-ph_SEOB[0])), label = '2,1 - SEOB')

#plt.plot(t_grid, np.abs(h_mlgw), label = '2,1')
#plt.plot(t_grid, np.abs(h_SEOB), label = '2,1 - SEOB')

plt.show()













