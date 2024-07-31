"""
Module GW_helper.py
===================
	Various routines for performing some tasks regarding GW signals:
		Mismatch computation
			function compute_mismatch: computes the mismatch between two sets of wave (useful for seeing how important is reconstruction error)
		Scalar product computation
			function compute_scalar: computes the Wigner scalar product between two GW waveforms
		Optimal mismatch computation:
			function compute_optimal_mismatch: computes the optimal mismatch between two waves (i.e. by minimizing the mismatch w.r.t. the alignment)
		Dataset creation Time Domain
			function create_dataset_TD: creates a dataset of GW in time domain.
		Dataset creation Frequency Domain
			function create_dataset_FD: creates a dataset of GW in frequency domain.
"""
#################

import numpy as np
import time
import os.path
import matplotlib.pyplot as plt #debug
import warnings
import scipy.signal
from tqdm import tqdm

################# Helpers with the frequency

def f_min(tau, q, M):
	"""
	Computes the approximate minimum frequency of a waveform, given the total mass, mass ratio and the length of the reduced time grid (s/M_sun)
	"""
	#return (151*(tau_min)**(-3./8.) * (((1+q)**2)/q)**(3./8.))/M
	return 151*np.power(np.square(1+q)/(q*np.abs(tau)), 3/8)/M

def f_ISCO(M):
	"""
	Computes the ISCO frequency for a system with a given total mass M
	"""
	M_SUN = 4.925490947641267e-06 #solar mass in seconds
	return 1/(6*np.sqrt(6)*2*np.pi*M_SUN*M)
	
def frequency22_merger(m1, m2, s1z, s2z): 
	""" 
	Computes the merger frequency defined with respect to the :math:`22` mode of the inspiral (Ref: Nagar)
	
	Input:
		m1		primary mass in units of solar masses, :math:`m_1\\geq m_2`
		m2		secondary mass in units of solar masses
		s1z		dimensionless spin of the primary mass, :math:`z` component
		s2z		dimensionless spin of the secondary mass, :math:`z` component
	
	Output:
		f_merger	the merger frequency of the 22 mode :math:`f_{22}`
	"""
	M_SUN = 4.925490947641267e-06 #solar mass in seconds
	
	q	   	= m1/m2  #Husa conventions, m1>m2 [https://arxiv.org/abs/1611.00332]
	eta	 	= q/(1+q)**2
	M_tot   = m1+m2
	chi_1   = 0.5*(1.0+np.sqrt(1.0-4.0*eta))
	chi_2   = 1.0-chi_1
	chi_eff = chi_1*s1z + chi_2*s2z
		
	A = -0.28562363*eta + 0.090355762
	B = -0.18527394*eta + 0.12596953
	C =  0.40527397*eta + 0.25864318
	
	res = (A*chi_eff**2 + B*chi_eff + C)*(2*np.pi*M_tot*M_SUN)**(-1)
	return res

################# Overlap related stuff

def overlap(amp_1, ph_1, amp_2, ph_2, df, low_freq = None, high_freq = None, PSD = None):
		#it works only for input shapes (D,)
	w1 = amp_1*np.exp(1j*ph_1)
	w2 = amp_2*np.exp(1j*ph_2)
	
	if PSD is None:
		PSD = np.ones((amp_1.shape[0],))
	if low_freq is not None and high_freq is not None:
		kmin, kmax = get_low_high_freq_index(low_freq, high_freq, df)
	else:
		kmin =0
		kmax = w1.shape[0]
	w1 = w1[kmin:kmax]
	w2 = w2[kmin:kmax]
	
	overlap = 2.0*df*np.sum(np.divide(np.conj(w1)*w2+np.conj(w2)*w1,PSD)).real
#	overlap = 4.0*df*np.sum(np.divide(np.conj(w1)*w2,PSD)).real
	return overlap

def compute_scalar(amp_1, ph_1, amp_2, ph_2, df, S = None):
	"""
compute_scalar
==============
	Computes an approximation to the Wigner scalar product for the two given GW.
		<h_1,h_2> = 4 Re[integral df h_1*(f) h_2(f)/S(f) ] ~ 4 Re sum[amp_1*(f_i) amp_2(f_i) exp(i*(ph_2(f_i)-ph_1(f_i)))/S(f_i) ] / D
	Input:
		amp_1/ph_1 (N,D)	amplitude/phase vector for wave 1 in Fourier space sampled in D uniform points within the domain
		amp_2/ph_2 (N,D)	amplitude/phase vector for wave 2 in Fourier space sampled in D uniform points within the domain
		df	()/(N,)			distance in the domain between two sampled points (can be uniform or different for each data point)
		S (N,D)				noise power spectral density sampled in D uniform points within the domain (if None there is no noise)
	Output:
		scalar (N,)	Wigner scalar product for the two waves
	"""
		#getting waves
	w1 = np.multiply(amp_1,np.exp(1j*ph_1))
	w2 = np.multiply(amp_2,np.exp(1j*ph_2))

		#checking for noise
	if S is None:
		S = np.ones(w1.shape)
	else:
		if S.shape != w1.shape:
			raise TypeError('Noise doesn\'t have the shape of data ('+str(amp_1.shape)+')')
			return None

	if w1.ndim < 2 and w2.ndim < 2:
		product = 4.0*np.multiply(df,np.sum(np.divide(np.conj(w1)*w2,S))).real
	else:	
		product = 4.0*np.multiply(df,np.sum(np.divide(np.conj(w1)*w2,S), axis = 1)).real
	
	return product.real


def compute_mismatch(amp_1, ph_1, amp_2, ph_2, S = None):
	"""
compute_mismatch
================
	Compute mismatch F between the waves given in input. Mismatch is computed with the formula
		F = 1-<h_1,h_2>/sqrt(<h_1,h_1><h_2,h_2>)
	with <,> being the Wigner scalar product for GW.
	Warning: waves must be aligned. Please use compute_optimal_mismatch for unaligned waves.
	Input:
		amp_1/ph_1 (N,D)	amplitude/phase vector for wave 1 in Fourier space sampled in D uniform points within the domain
		amp_2/ph_2 (N,D)	amplitude/phase vector for wave 1 in Fourier space sampled in D uniform points within the domain
		S (D,)				noise power spectral density sampled in D uniform points within the domain (if None there is no noise)
	Output:
		F (N,)	Mismatch between waves computed element-wise (i.e. F[i] holds mismatch between h_1[i,:] and h_2[i,:])
	"""
		#checking if data make sense
	if len({amp_1.shape, ph_1.shape, amp_2.shape, ph_2.shape}) == 1: #python set contains unique values
		if amp_1.ndim == 1:
			amp_1 = np.reshape(amp_1, (1,amp_1.shape[0]))
			amp_2 = np.reshape(amp_2, (1,amp_2.shape[0]))
			ph_1 = np.reshape(ph_1, (1,ph_1.shape[0]))
			ph_2 = np.reshape(ph_2, (1,ph_2.shape[0]))
		D = amp_1.shape[1]
		N = amp_1.shape[0]
	else:
		raise TypeError('Data don\'t have the same shape')
		return None

		#computing mismatch vector
	df = 1. #nothing depends on df so we can set it arbitrarly
	F = compute_scalar(amp_1, ph_1, amp_2, ph_2, df, S)
	div_factor = np.sqrt(np.multiply(compute_scalar(amp_2, ph_2, amp_2, ph_2, df, S), compute_scalar(amp_1, ph_1, amp_1, ph_1, df, S)))
	np.divide(F, div_factor, out = F)
	return 1-F

def compute_optimal_mismatch(h1,h2, optimal = True, return_F = True):
	"""
compute_optimal_mismatch
========================
	Computes the optimal mismatch/overlap between two complex waveforms by performing the minimization:
		F = min_phi F[h1, h2*exp(1j*phi)]
	After the computation, h1 and h2*exp(1j*phi) are optimally aligned.
	Input:
		h1 (N,D)/(D,)	complex wave
		h2 (N,D)/(D,)	complex wave
		optimal			whether to optimize w.r.t. a constant phase
		return_F		whether to reteurn mismatch or overlap
	Output:
		F_optimal (N,)/()		optimal mismatch/overlap
		phi_optimal (N,)/()		optimal phi which aligns the two waves
	"""
	assert h1.shape == h2.shape
	if h1.ndim == 1:
		h1 = h1[np.newaxis,:] #(1,D)
		h2 = h2[np.newaxis,:] #(1,D)

	scalar = lambda h1_, h2_: np.sum(np.multiply(h1_,np.conj(h2_)), axis = 1)/h1_.shape[1] #(N,)

	norm_factor = np.sqrt(np.multiply(scalar(h1,h1).real, scalar(h2,h2).real))
	if optimal:
		overlap = scalar(h1,h2) #(N,)
		phi_optimal = np.angle(overlap) #(N,)
	else:
		phi_optimal = np.zeros(norm_factor.shape)
	overlap = np.divide(scalar(h1,(h2.T*np.exp(1j*phi_optimal)).T), norm_factor)
	overlap = overlap.real
	
	if return_F:
		return 1-overlap, phi_optimal
	else:
		return overlap, phi_optimal



################# Dataset related stuff

#@profile
def create_dataset_TD(N_data, N_grid, modes, basefilename,  t_coal = 0.5, q_range = (1.,5.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8), t_step = 1e-5, alpha = 0.35, approximant = "SEOBNRv2_opt", path_TEOBResumS = None):
	"""
create_dataset_TD
=================
	Create datasets for training a ML model to fit GW modes in time domain. Each dataset is saved in a different file (basefilename.mode).
	The dataset consists in 3 parameters theta=(q, spin1z, spin2z) associated to the modes computed in time domain for a grid of N_grid points in the range given by the user.
	More specifically, data are stored in 3 vectors:
		theta_vector	vector holding source parameters q, spin1, spin2
		amp_vector		vector holding amplitudes for each source evaluated at some discrete grid points
		ph_vector		vector holding phase for each source evaluated at some discrete grid points
	This routine add N_data data to filename if one is specified (if file is not empty it must contain data with the same N_grid); otherwise the datasets are returned as np vectors. 
	Values of q and m2 as well as spins are drawn randomly in the range given by the user: it holds m1 = q *m2 M_sun.
	The waveforms are computed with a time step t_step; starting from a time in reduced grid tau min (s/M_Sun). Waves are given in a rescaled time grid (i.e. t/m_tot) with N_grid points: t=0 occurs when the 22 mode has a peak. A higher density of grid points is placed close to the merger.
	Dataset is generated either with an implementation of TEOBResumS (a path to a local installation of TEOBResumS should be provided) either with SEOBNRv4HM (lalsuite installation required). It can be given an TD lal approximant with no HM; in this case, only the 22 mode can be generated.
	Datasets can be loaded with load_dataset.
	Input:
		N_data				size of dataset
		N_grid				number of grid points to evaluate
		modes []			list of modes (each a (l,m) tuple) to generate and fill the dataset with				
		basefilename		basename of the file to save dataset in (each dataset is saved in basefilename.lm)
		t_coal				time to coalescence to start computation from (measured in reduced grid)
		q_range				tuple with range for random q values. if single value, q is kept fixed at that value
		m2_range			tuple with range for random m2 values. if single value, m2 is kept fixed at that value. If None, m2 will be chosen s.t. m_tot = m1+m2 = 20. M_sun
		spin_mag_max_1		tuple with range for random spin #1 values. if single value, s1 is kept fixed at that value
		spin_mag_max_2		tuple with range for random spin #1 values. if single value, s2 is kept fixed at that value
		t_step				time step to generate the wave with
		approximant			string for the approximant model to be used (in lal convention; to be used only if lal is to be used)
		alpha				distorsion factor for time grid. (In range (0,1], when it's close to 0, more grid points are around merger)
		approximant			lal approximant to be used for generating the modes, or "TEOBResumS" (in the latter case a local installation must be provided by the argument path_TEOBResumS) 
		path_TEOBResumS		path to a local installation of TEOBResumS with routine 'EOBRun_module' (has effect only if approximant is "TEOBResumS")
	"""
	#TODO: create a function get modes, wrapper to ChooseTDModes: you can call it from here to get the modes
	if isinstance(modes,tuple):
		modes = [modes]
	if not isinstance(modes,list):
		raise TypeError("Wrong kind of mode {} given. Expected a list [(l,m)]".format(modes))

	if approximant == "TEOBResumS":
		#see https://bitbucket.org/eob_ihes/teobresums/src/development/ for the implementation of TEOBResumS
		if not isinstance(path_TEOBResumS, str):
			raise ValueError("Missing path to TEOBResumS: unable to continue")
		try:
			import sys
			sys.path.append(path_TEOBResumS) #path to local installation of TEOBResumS
			import EOBRun_module
		except:
			raise RuntimeError("No valid imput source for module 'EOBRun_module' for TEOBResumS. Unable to continue.")
	else:
		try:
			import lal
			import lalsimulation as lalsim
		except:
			raise RuntimeError("Impossible to load lal and lalsimulation: try pip install lalsuite")
		LALpars = lal.CreateDict()
		approx = lalsim.SimInspiralGetApproximantFromString(approximant)
		prefactor = 4.7864188273360336e-20 # G/c^2*(M_sun/Mpc)

			#checking that all is good with modes
		if approximant in ["SEOBNRv4PHM", "SEOBNRv4HM"]:
			for mode in modes:
				if mode not in [(2,2),(2,1), (3,3), (4,4), (5,5)]:
					raise ValueError("Currently SEOBNRv4PHM approximant does not implement the chosen HMs")
		elif approximant != "IMRPhenomTPHM":
			if modes != [(2,2)]:
				raise ValueError("The chosen lal approximant does not implement HMs")

		

		#checking if N_grid is fine
	if not isinstance(N_grid, int):
		raise TypeError("N_grid is "+str(type(N_grid))+"! Expected to be a int.")

	if isinstance(m2_range, tuple):
		D_theta = 4 #m2 must be included as a feature
	else:
		D_theta = 3

		######setting the time grid
	time_grid_list = []
	t_end_list = []
	if approximant == "TEOBResumS":
		modes_to_k = lambda modes:[int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes] # [(l,m)] -> [k]
		k_modes = modes_to_k(modes)
			#setting a list of time grids
		for mode in modes:
				#ugly setting of t_end in TEOBResumS: required to kill bad features after merger
			if mode == (2,2):
				t_end = 5.2e-4 #estimated maximum time for ringdown: WF will be killed after that time
			elif mode == (2,1) or mode == (3,3): #special treatment for 21 and 33
				t_end = 1e-6
			else:
				t_end = 3e-5 #for other modes
			t_end_list.append(t_end)
	else:
		for mode in modes:
			t_end_list.append(5.2e-4)

	print("Generating modes: "+str(modes))

		#creating time_grid
	for i,mode in enumerate(modes):
		time_grid = np.linspace(-np.power(np.abs(t_coal), alpha), np.power(t_end_list[i], alpha), N_grid)
		time_grid = np.multiply( np.sign(time_grid) , np.power(np.abs(time_grid), 1./alpha))

			#adding 0 to time grid
		index_0 = np.argmin(np.abs(time_grid))
		time_grid[index_0] = 0. #0 is alway set in the grid

		time_grid_list.append(time_grid)

		#setting t_coal_freq for generating a waves
	if np.abs(t_coal) < 0.05:
		t_coal_freq = 0.05
	else:
		t_coal_freq = np.abs(t_coal)


		#####create a list of buffer to save the WFs
	buff_list = []
	for i, mode in enumerate(modes):
		filename = basefilename+'.'+str(mode[0])+str(mode[1])
		if not os.path.isfile(filename): #file doesn't exist: must be created with proper header
			filebuff = open(filename,'w')
			print("New file ", filename, " created")
			time_header = np.concatenate((np.zeros((D_theta,)), time_grid_list[i], time_grid_list[i]) )[None,:]
			np.savetxt(filebuff, time_header, header = "#Mode:"+ str(mode[0])+str(mode[1]) +"\n# row: theta "+str(D_theta)+" | amp (None,"+str(N_grid)+")| ph (None,"+str(N_grid)+")\n# N_grid = "+str(N_grid)+" | t_coal ="+str(t_coal)+" | t_step ="+str(t_step)+" | q_range = "+str(q_range)+" | m2_range = "+str(m2_range)+" | s1_range = "+str(s1_range)+" | s2_range = "+str(s2_range), newline = '\n')
		else:
			filebuff = open(filename,'a')
		buff_list.append(filebuff)

		#####creating WFs
	for n_WF in tqdm(range(N_data), desc = 'Generating dataset'):
			#setting value for data
		if isinstance(m2_range, (tuple, list)):
			m2 = np.random.uniform(m2_range[0],m2_range[1])
		elif m2_range is not None:
			m2 = float(m2_range)
		if isinstance(q_range, (tuple, list)):
			#q = np.random.uniform(q_range[0],q_range[1]) #uniform q distribution
			
			#biased q distribution for the boundaries
			x = np.random.uniform()
			if x < 0.3:
				q = np.min(np.random.uniform(low=q_range[0],high=q_range[1],size=5))
			elif 0.3 <= x < 0.8:
				q = np.random.uniform(low=q_range[0],high=q_range[1])
			else:
				q = np.max(np.random.uniform(low=q_range[0],high=q_range[1],size=5))
		else:
			q = float(q_range)
		if isinstance(s1_range, (tuple, list)):
			spin1z = np.random.uniform(s1_range[0],s1_range[1])
		else:
			spin1z = float(s1_range)
		if isinstance(s2_range, (tuple, list)):
			spin2z = np.random.uniform(s2_range[0],s2_range[1])
		else:
			spin2z = float(s2_range)

		if m2_range is None:
			m2 = 20. / (1+q)
			m1 = q * m2
		else:
			m1 = q* m2
		nu = np.divide(q, np.square(1+q)) #symmetric mass ratio

			#computing f_min
		#f_min = .9* ((151*(t_coal_freq)**(-3./8.) * (((1+q)**2)/q)**(3./8.))/(m1+m2))
		f_min = .9*f_min(t_coal_freq, q, m1+m2)
		 #f_min is the right scaling formula for frequency in order to get always the right reduced time
		 #this should be multiplied by a prefactor (~1) for dealing with some small variation due to spins

		if isinstance(m2_range, tuple):
			temp_theta = [m1, m2, spin1z, spin2z]		
		else:
			temp_theta = [m1/m2, spin1z, spin2z]

			#getting the waveform
			#output of the if:
				#amp_list, ph_list (same order as in modes)
				#time_full, argpeak
		amp_list, ph_list = [None for i in range(len(modes))],[None for i in range(len(modes))]
		if approximant == "TEOBResumS": #using TEOBResumS
			pars = {
				'M'                  : m1+m2,
				'q'                  : m1/m2,
				'Lambda1'            : 0.,
				'Lambda2'            : 0.,     
				'chi1'               : spin1z,
				'chi2'               : spin2z,
				'domain'             : 0,      # TD
				'arg_out'            : 1,      # Output hlm/hflm. Default = 0
				'use_mode_lm'        : list(set(k_modes + [1])),      # List of modes to use/output through EOBRunPy (added 22 mode in case there isn't)
				#'srate_interp'       : 1./t_step,  # srate at which to interpolate. Default = 4096.
				'use_geometric_units': 0,      # Output quantities in geometric units. Default = 1
				'initial_frequency'  : f_min,   # in Hz if use_geometric_units = 0, else in geometric units
				'interp_uniform_grid': 0,      # Interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
				'distance': 1.,
				'inclination':0.,
			}
			time_full, h_p, h_c, hlm = EOBRun_module.EOBRunPy(pars)
			for i, k_mode in enumerate(k_modes):
				temp_amp = hlm[str(k_mode)][0]*nu #TEOBResumS has weird conventions on the modes
				temp_ph = hlm[str(k_mode)][1]
				amp_list[i] = temp_amp
				ph_list[i] = temp_ph
			argpeak = locate_peak(hlm['1'][0]*nu) #aligned at the peak of the 22

		elif approximant == "SEOBNRv4HM": #using SEOBNRv4HM
			warnings.warn("The dataset generation with SEOBNRv4HM has not been extensively tested!")
			nqcCoeffsInput=lal.CreateREAL8Vector(10)
			sp, dyn, dynHi = lalsim.SimIMRSpinAlignedEOBModes(t_step, m1*lal.MSUN_SI, m2*lal.MSUN_SI, f_min, 1e6*lal.PC_SI, spin1z, spin2z,41, 0., 0., 0.,0.,0.,0.,0.,0.,1.,1.,nqcCoeffsInput, 0)
			amp_prefactor = prefactor*(m1+m2)/1. # G/c^2 (M / d_L)
			while sp is not None:
				lm = (sp.l, sp.m)
				if lm not in modes: #skipping a non-necessary mode
					continue
				else: #computing index and saving the mode
					i = modes.index(lm)
					hlm = sp.mode.data.data #complex mode vector
					temp_amp = np.abs(hlm)/ amp_prefactor / nu #scaling with the convention of SEOB
					temp_ph = np.unwrap(np.angle(hlm))
					amp_list[i] = temp_amp
					ph_list[i] = temp_ph

				if (sp.l, sp.m) == (2,2): #get grid
					amp_22 = np.abs(sp.mode.data.data) #amp of 22 mode (for time grid alignment)
					time_full = np.linspace(0.0, sp.mode.data.length*t_step, sp.mode.data.length) #time grid at which wave is computed
					argpeak = locate_peak(amp_22) #aligned at the peak of the 22
				sp = sp.next
		elif approximant == "IMRPhenomTPHM" or approximant == "SEOBNRv4PHM":
			#https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/test___s_e_o_b_n_rv4_p_h_m__vs__4_h_m__ringdown_8py_source.html
			approx = lalsim.SEOBNRv4PHM if approximant == "SEOBNRv4PHM" else lalsim.IMRPhenomTPHM
			try:
				hlm = lalsim.SimInspiralChooseTDModes(0.,
					t_step,
					m1*lalsim.lal.MSUN_SI,
					m2*lalsim.lal.MSUN_SI,
					0.,
					0.,
					spin1z,
					0.,
					0.,
					spin2z,
					f_min,
					f_min,
					1e6*lalsim.lal.PC_SI,
					LALpars,
					5,			#lmax
					approx
				)
			except RuntimeError:
				print("Unable to generate WF @ theta = {}".format(temp_theta))
				continue
			amp_prefactor = prefactor*(m1+m2)/1.
			for i, lm in enumerate(modes):
				#{(2,2): lal.ts} 
				temp_hlm = np.array(lalsim.SphHarmTimeSeriesGetMode(hlm, lm[0], lm[1]).data.data)
				temp_amp = np.abs(temp_hlm)/ amp_prefactor / nu #check that this conventions are for every lal part
				temp_ph = np.unwrap(np.angle(temp_hlm))
				amp_list[i] = temp_amp
				ph_list[i] = temp_ph
				if (lm[0], lm[1]) == (2,2): #get grid
					argpeak = locate_peak(temp_amp) #aligned at the peak of the 22
			time_full = np.linspace(0.0, len(temp_amp)*t_step, len(temp_amp)) #time grid at which wave is computed
		else: #another lal approximant (only 22 mode)
			hp, hc = lalsim.SimInspiralChooseTDWaveform( #where is its definition and documentation????
				m1*lalsim.lal.MSUN_SI, #m1
				m2*lalsim.lal.MSUN_SI, #m2
				0., 0., spin1z, #spin vector 1
				0., 0., spin2z, #spin vector 2
				1e6*lalsim.lal.PC_SI, #distance to source
				0., #inclination
				0., #phi
				0., #longAscNodes
				0., #eccentricity
				0., #meanPerAno
				t_step, # time incremental step
				f_min, # lowest value of freq
				f_min, #some reference value of freq (??)
				LALpars, #some lal dictionary
				approx #approx method for the model
			)
			h_p, h_c =  hp.data.data, hc.data.data
			time_full = np.linspace(0.0, hp.data.length*t_step, hp.data.length) #time grid at which wave is computed
			amp_prefactor = prefactor*(m1+m2)/1. # G/c^2 (M / d_L)
			temp_amp = np.sqrt(np.square(h_p)+np.square(h_c)) / amp_prefactor / (4*np.sqrt(5/(64*np.pi)))
			temp_ph = np.unwrap(np.arctan2(h_c,h_p))
			amp_list = [temp_amp]
			ph_list = [temp_ph]
			argpeak = locate_peak(temp_amp) #aligned at the peak of the 22
			hlm = None

			#setting time grid
		t_peak = time_full[argpeak]
		time_full = (time_full - t_peak)/(m1+m2) #grid is scaled to standard grid

			#computing waves to the chosen std grid and saving to file
		for i in range(len(amp_list)):
			temp_amp, temp_ph = amp_list[i], ph_list[i]
			temp_amp = np.interp(time_grid_list[i], time_full, temp_amp)
			temp_ph = np.interp(time_grid_list[i], time_full, temp_ph)
			temp_ph = temp_ph - temp_ph[0] #all phases are shifted by a constant to make sure every wave has 0 phase at beginning of grid

			if False:
				#TODO: remove this shit!!
				plt.figure()
				plt.plot(time_grid_list[i], temp_amp_,'o', ms = 2)
				plt.plot(time_full, temp_amp)
				plt.xlim([-0.00015,0.00015])
				plt.show()

			to_save = np.concatenate((temp_theta, temp_amp, temp_ph))[None,:] #(1,D)
			np.savetxt(buff_list[i], to_save)
	
		del temp_theta, temp_amp, temp_ph
		del amp_list, ph_list
		del to_save
		del hlm
			

			#user communication
		#if n_WF%50 == 0 and n_WF != 0:
		#if n_WF%1 == 0 and n_WF != 0: #debug
		#	print("Generated WF ", n_WF)

	filebuff.close()
	return

def generate_mode(m1,m2, s1=0.,s2 = 0., d=1., t_coal = 0.4, t_step = 5e-5, f_min = None, t_min = None, verbose = False, approx = "IMRPhenomTPHM"):
	"""
generate_mode
=============
	Wrapper to lalsimulation.SimInspiralChooseTDModes() to generate modes of a waveform. Wave is not preprocessed.
	Input:
		m1,m2,s1,s2,d				orbital parameters
		t_coal						approximate time to coalescence in reduced grid (ignored if f_min or t_min is set)
		t_step						EOB integration time to be given to lal
		f_min						starting frequency in Hz (if None, it will be determined by t_coal; ignored if t_min is set)
		t_min						starting time in s (if None, t_coal will be returned)
		verbose						whether to print messages for each wave...
	Output:
		times (D,)	times at which modes are evaluated
		mode_dict	dictionary with the modes (each mode is a complex timeseries)
	"""
	import lal
	import lalsimulation as lalsim
	q = m1/m2
	mtot = (m1+m2)#*lal.MTSUN_SI
	mc = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
	mc /= 1.21 #M_c / 1.21 M_sun

	if t_min is not None:
		f_min = .8* ((151*(t_min/mtot)**(-3./8.) * (((1+q)**2)/q)**(3./8.))/mtot)

	if f_min is None:
		f_min = .9* ((151*(t_coal)**(-3./8.) * (((1+q)**2)/q)**(3./8.))/mtot)

	if verbose:
		print("Generating wave @: ",m1,m2,s1,s2,d, iota, phi_0)

	hlm = lalsim.SimInspiralChooseTDModes(0.,
				t_step,
				m1*lalsim.lal.MSUN_SI,
				m2*lalsim.lal.MSUN_SI,
				0.,
				0.,
				s1z,
				0.,
				0.,
				s2z,
				f_min,
				f_min,
				1e6*d*lalsim.lal.PC_SI,
				LALpars,
				5,			#lmax
				lalsim.GetApproximantFromString(approx) #approx method for the model
			)
		
	mode_dict = {}
	for l in range(2,6):
		for m in range(1,l+1):
			#{(2,2): lal.ts}
			try:
				temp_hlm = lalsim.SphHarmTimeSeriesGetMode(hlm, lm[0], lm[1]).data.data
				mode_dict[(l,m)] = np.array(temp_hlm)
			except:
				continue

	amp = np.abs(mode_dict[(2,2)])
	times = np.linspace(0.0, amp.shape[0]*t_step, amp.shape[0])  #time actually
	t_m =  times[np.argmax(amp)]
	times = times - t_m

	if t_min is not None:
		arg = np.argmin(np.abs(times+t_min))
		for k, v in mode_dict.items():
			mode_dict[k] = v[arg:]
		times = times[arg:]

	return times, mode_dict


def generate_waveform(m1,m2, s1=0.,s2 = 0.,d=1., iota = 0.,phi_0=0., t_coal = 0.4, t_step = 5e-5, f_min = None, t_min = None, verbose = False, approx = "SEOBNRv2_opt"):
	"""
generate_waveform
=================
	Wrapper to lalsimulation.SimInspiralChooseTDWaveform() to generate a single waveform. Wave is not preprocessed.
	Input:
		m1,m2,s1,s2,d,iota,phi_0	orbital parameters
		t_coal						approximate time to coalescence in reduced grid (ignored if f_min or t_min is set)
		t_step						EOB integration time to be given to lal
		f_min						starting frequency in Hz (if None, it will be determined by t_coal; ignored if t_min is set)
		t_min						starting time in s (if None, t_coal will be returned)
		verbose						whether to print messages for each wave...
	Output:
		times (D,)	times at which wave is evaluated
		h_p (N,D)	plus polarization of the wave
		h_c (N,D)	cross polarization of the wave
	"""
	import lal
	import lalsimulation as lalsim
	q = m1/m2
	mtot = (m1+m2)#*lal.MTSUN_SI
	mc = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
	mc /= 1.21 #M_c / 1.21 M_sun

	if t_min is not None:
		f_min = .8* ((151*(t_min/mtot)**(-3./8.) * (((1+q)**2)/q)**(3./8.))/mtot)

	if f_min is None:
		f_min = .9* ((151*(t_coal)**(-3./8.) * (((1+q)**2)/q)**(3./8.))/mtot)

	if verbose:
		print("Generating wave @: ",m1,m2,s1,s2,d, iota, phi_0)

	hp, hc = lalsim.SimInspiralChooseTDWaveform( #where is its definition and documentation????
		m1*lalsim.lal.MSUN_SI, #m1
		m2*lalsim.lal.MSUN_SI, #m2
		0., 0., s1, #spin vector 1
		0., 0., s2, #spin vector 2
		d*1e6*lalsim.lal.PC_SI, #distance to source
		iota, #inclination
		phi_0, #phi
		0., #longAscNodes
		0., #eccentricity
		0., #meanPerAno
		t_step, # time incremental step
		f_min, # lowest value of freq
		f_min, #some reference value of freq (??)
		lal.CreateDict(), #some lal dictionary
		lalsim.GetApproximantFromString(approx) #approx method for the model
		)
	h_p, h_c =  hp.data.data, hc.data.data
	amp = np.sqrt(h_p**2+h_c**2)
	#(indices, ) = np.where(amp!=0) #trimming zeros of amplitude
	#h_p = h_p[indices]
	#h_c = h_c[indices]

	times = np.linspace(0.0, h_p.shape[0]*t_step, h_p.shape[0])  #time actually
	t_m =  times[np.argmax(amp)]
	times = times - t_m

	if t_min is not None:
		arg = np.argmin(np.abs(times+t_min))
	else:
		arg=0

	return times[arg:], h_p[arg:], h_c[arg:]

def generate_waveform_TEOBResumS(m1,m2, s1=0.,s2 = 0.,d=1., iota = 0., t_coal = 0.4, t_step = 5e-5, f_min = None, t_min = None,
modes = [(2,2)], verbose = False, path_TEOBResumS = None):
	"""
generate_waveform
=================
	Wrapper to EOBRun_module.EOBRunPy() to generate a single waveform with TEOBResumS. Wave is not preprocessed.
	Input:
		m1,m2,s1,s2,d,iota,phi_0	orbital parameters
		t_coal						approximate time to coalescence in reduced grid (ignored if f_min or t_min is set)
		t_step						EOB integration time to be given to lal
		f_min						starting frequency in Hz (if None, it will be determined by t_coal; ignored if t_min is set)
		t_min						starting time in s (if None, t_coal will be returned)
		modes 						list of modes to be included in the WFs
		verbose						whether to print messages for each wave...
	Output:
		times (D,)	times at which wave is evaluated
		h_p (D,)	plus polarization of the wave
		h_c (D,)	cross polarization of the wave
		t_m 		time at which amplitude peaks
	"""
	modes_to_k = lambda modes:[int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes] # [(l,m)] -> [k]

	if path_TEOBResumS is None: #very ugly but useful
		path_TEOBResumS = '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/TEOBResumS/Python'
	import sys
	sys.path.append(path_TEOBResumS) #path to local installation of TEOBResumS
	import EOBRun_module

	q = m1/m2
	if q <1:
		q = m2/m1
		s1, s2 = s2, s1
	mtot = (m1+m2)#*lal.MTSUN_SI
	mc = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
	mc /= 1.21 #M_c / 1.21 M_sun

	if t_min is not None:
		f_min = .8* ((151*(t_min/mtot)**(-3./8.) * (((1+q)**2)/q)**(3./8.))/mtot)

	if f_min is None:
		f_min = .9* ((151*(t_coal)**(-3./8.) * (((1+q)**2)/q)**(3./8.))/mtot)

	if verbose:
		print("Generating wave @: ",m1,m2,s1,s2,d, iota)
	
	k_modes = modes_to_k(modes)

	pars = {'M'                  : m1+m2,
			'q'                  : m1/m2,
			'Lambda1'            : 0.,
			'Lambda2'            : 0.,     
			'chi1'               : s1,
			'chi2'               : s2,
			'domain'             : 0,      # TD
			'arg_out'            : 1,     # Output hlm/hflm. Default = 0
			'use_mode_lm'        : k_modes,      # List of modes to use/output through EOBRunPy
			'srate_interp'       : 1./t_step,  # srate at which to interpolate. Default = 4096.
			'use_geometric_units': 0,      # Output quantities in geometric units. Default = 1
			'initial_frequency'  : f_min,   # in Hz if use_geometric_units = 0, else in geometric units
			'interp_uniform_grid': 2,      # Interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
			'distance': d,
			'inclination': iota,
			'nqc':1, #{"no", "auto", "manual"}
			'nqc_coefs_flx':0, # {"none", "nrfit_nospin20160209", "fit_spin_202002", "fromfile"}
			'nqc_coefs_hlm':0
		}
	times, h_p, h_c, hlm = EOBRun_module.EOBRunPy(pars)

	try:
		argpeak = locate_peak(hlm['1'][0])
	except:
		amp = np.sqrt(h_p**2+h_c**2)
		argpeak = locate_peak(amp)
	t_m =  times[argpeak]
	times = times - t_m

	if t_min is not None:
		arg = np.argmin(np.abs(times+t_min))
	else:
		arg=0

	return times[arg:], h_p[arg:], h_c[arg:], hlm, t_m


def locate_peak(amp, start = 0.3):
	"""
	Given a time grid and an amplitude of the mode, it returns the peak of the amplitude.
	Input:
		amp (D,)		amplitude
		start			points to be skipped at begining of the amplitude (in fraction of total points)
	Output:
		argpeak			index at which the amplitude has a peak
	"""
	assert amp.ndim == 1
	id_start = int(len(amp)*start)
	extrema = scipy.signal.argrelextrema(np.abs(amp[id_start:]), np.greater)
	if len(extrema[0]):
		return extrema[0][0]+id_start
	else:
		return np.argmax(np.abs(amp[id_start:]))+id_start



def save_dataset(filename, theta_vector, amp_dataset, ph_dataset, x_grid):
	"""
	Save a dataset in a way that it's readable by load_dataset.
	Input:
		filename	name of the file to save dataset to
		theta_vector (N_data,3)		vector holding ordered set of parameters used to generate amp_dataset and ph_dataset
		amp_dataset (N_data,N_grid)	dataset with amplitudes
		ph_dataset (N_data,N_grid)	dataset with phases
		x_grid (N_grid,)			vector holding x_grid at which waves are evaluated
	"""
	to_save = np.concatenate((theta_vector, amp_dataset, ph_dataset), axis = 1)
	temp_x_grid = np.zeros((1,to_save.shape[1]))
	K = int((to_save.shape[1]-3)/2)
	temp_x_grid[0,3:3+k] = x_grid
	to_save = np.concatenate((temp_x_grid,to_save), axis = 0)
	q_max = np.max(theta_vector[:,0])
	spin_mag_max = np.max(np.abs(theta_vector[:,1:2]))
	x_step = x_grid[1]-x_grid[0]
	np.savetxt(filename, to_save, header = "# row: theta 3 | amp "+str(amp_dataset.shape[1])+"| ph "+str(ph_dataset.shape[1])+"\n# N_grid = "+str(x_grid.shape[0])+" | f_step ="+str(x_step)+" | q_max = "+str(q_max)+" | spin_mag_max = "+str(spin_mag_max), newline = '\n')
	return


def load_dataset(filename, N_data=None, N_entries = 2, N_grid = None, shuffle = False, n_params = 3):
	"""
	Load a dataset from file. The file should be suitable for np arrays and have the following structure:
		parameters n_params | entry_1 K | entry_2 K | ... | entry_2 D
	It is suitable for GW dataset (n_params = 3 and entry_1/2 = amplitude/phase), for angles dataset (n_params = 6 and entry_1/2 = alpha/beta) and for "extended" angle dataset (n_params = 6 and entries = alpha, beta_m, beta_amp, beta_ph)
	The first row hold the x_grid vector.
	It can shuffle the data if required.
	Input:
		filename	input filename
		N_data		number of data to extract (only if data in file are more than N_data) (if None N_data = N)
		N_grid		number of grid points to evaluate the waves in (Only if N_grid < N_grid_dataset)
		shuffle		whether to shuffle data
		n_params	number of columns in the theta_vector
	Outuput:
		A list storing the following np.array
			theta_vector (N_data,n_params)	vector holding ordered set of parameters used to generate amp_dataset and ph_dataset
			dataset_1 (N_data,K)			entry 1
			.
			.
			dataset_D	(N_data,K)			entry D
			x_grid (K,)						vector holding x_grid at which the entries are evaluated (can be frequency or time grid)
	"""
		#here entry1 is amp and entry2 is ph (change it)
	if N_data is not None:
		N_data += 1
	data = np.loadtxt(filename, max_rows = N_data)
	N = data.shape[0]
	D = N_entries
	K = int((data.shape[1]-n_params)/D)
	x_grid = data[0,n_params:n_params+K] #saving x_grid

	if data.shape[1] != D*K + n_params:
		raise ValueError("File given is not suitable for the required dataset. Unable to continue")

	data = data[1:,:] #removing x_grid from full data
	if shuffle: 	#shuffling if required
		np.random.shuffle(data)

	theta_vector = data[:,0:n_params]
	dataset_list = []
	for d in range(D):
		dataset_list.append(data[:,n_params+d*K:n_params+(d+1)*K])

	if N_grid is not None:
		if dataset_list[0].shape[1] < N_grid:
			warnings.warn("Not enough grid points ("+str(dataset_list[0].shape[1])+") for the required N_grid value ("+str(N_grid)+").\nMaximum number of grid point is taken (but less than N_grid)")
			N_grid = dataset_list[0].shape[1]
		indices = np.random.choice(range(dataset_list[0].shape[1]), size = (N_grid,), replace = False)
		x_grid = x_grid[indices]
		for d in range(D):
			dataset_list[d] = dataset_list[d][:,indices]

	to_return = [theta_vector, *dataset_list, x_grid]

	return to_return

def make_set_split(data, labels, train_fraction = .85, scale_factor = None):
	"""
	Given a GW dataset made of data and labels, it makes a split between training and test set. Labels are scaled for scale factor (labels = labels/scale_factor).
	Input:
		data (N,K)		parameters set
		labels (N,L)	label set
		train_fraction	the fraction of data to included in training set
		scale_factor	scale factor for scaling data (if None data are not scaled)
	Output:
		train_data/test_data (N*train_frac/N*(1-train_frac),K)	parameters for training/test set
		train_labels/test_labels (N*train_frac/N*(1-train_frac),K)	labels for training/test set
	"""
	if scale_factor is None:
		scale_factor = 1.
	labels = labels/scale_factor

		#splitting into train and test set
	N = data.shape[0]
	train_data = np.array(data[:int(train_fraction*N),:])
	test_data = np.array(data[int(train_fraction*N):,:])

	train_labels = np.array(labels[:int(train_fraction*N),:])
	test_labels = np.array(labels[int(train_fraction*N):,:])
	return train_data, test_data, train_labels, test_labels

