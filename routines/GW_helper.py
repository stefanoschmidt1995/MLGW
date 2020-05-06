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
import os.path
import matplotlib.pyplot as plt #debug

################# Overlap related stuff
def get_low_high_freq_index(flow, fhigh, df):
	kmin = int(flow / df)
	kmax = int(fhigh / df)
	return kmin, kmax

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

def compute_optimal_mismatch(h1,h2, return_F = True):
	"""
compute_optimal_mismatch
========================
	Computes the optimal mismatch/overlap between two complex waveforms by performing the minimization:
		F = min_phi F[h1, h2*exp(1j*phi)]
	After the computation, h1 and h2*exp(1j*phi) are optimally aligned.
	Input:
		h1 (N,D)/(D,)	complex wave
		h2 (N,D)/(D,)	complex wave
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
	overlap = scalar(h1,h2) #(N,)
	phi_optimal = np.angle(overlap) #(N,)
	overlap = np.divide(scalar(h1,h2*np.exp(1j*phi_optimal)), norm_factor)
	overlap = overlap.real

		#debug
	#plt.plot(np.linspace(0,h1.shape[1],h1.shape[1]),np.squeeze(h1))
	#plt.plot(np.linspace(0,h1.shape[1],h1.shape[1]),np.squeeze(h2)*np.exp(1j*phi_optimal))
	#print(1-overlap, overlap)
	#plt.show()


	if return_F:
		return 1-overlap, phi_optimal
	if not return_F:
		return overlap, phi_optimal



################# Dataset related stuff

def create_dataset_TD(N_data, N_grid, filename = None,  t_coal = 0.5, q_range = (1.,5.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8), t_step = 1e-5, approximant = "SEOBNRv2_opt", alpha = 0.35, path_TEOBResumS = None):
	"""
create_dataset_TD
=================
	Create a dataset for training a ML model to fit GW waveforms in time domain.
	The dataset consists in 3 parameters theta=(q, spin1z, spin2z) associated to the waveform computed in frequency domain for a grid of N_grid points in the range given by the user.
	More specifically, data are stored in 3 vectors:
		theta_vector	vector holding source parameters q, spin1, spin2
		amp_vector		vector holding amplitudes for each source evaluated at some N_grid equally spaced points
		ph_vector		vector holding phase for each source evaluated at some N_grid equally spaced points
	This routine add N_data data to filename if one is specified (if file is not empty it must contain data with the same N_grid); otherwise the datasets are returned as np vectors. 
	All the waves are evaluated at a constant distance of 1Mpc. Values of q and m2 as well as spins are drawn randomly in the range given by the user: it holds m1 = q *m2 M_sun.
	The waveforms are computed with a time step t_step; starting from a frequency f_min (set by the routine according to t_coal and m_tot). Waves are given in a rescaled time grid (i.e. t/m_tot) with N_grid points: t=0 occurs when at time of maximum amplitude. A higher density of grid points is placed in the post merger phase.
	Dataset can be generated either with a lal method (the approximant should be specified by the approximant keyword) either with an implementation of TEOBResumS (in this case a path to a local installation of TEOBResumS should be provided). If lal is used, lalsuite package shall be installed (note that lalsuite is not a prerequisite for mlgw)
	Dataset can be loaded with load_dataset.
	Input:
		N_data				size of dataset
		N_grid				number of grid points to evaluate
		filename			name of the file to save dataset in (If is None, nothing is saved on a file)
		t_coal				time to coalescence to start computation from (measured in reduced grid)
		q_range				tuple with range for random q values. if single value, q is kept fixed at that value
		m2_range			tuple with range for random m2 values. if single value, m2 is kept fixed at that value. If None, m2 will be chosen s.t. m_tot = m1+m2 = 20. M_sun
		spin_mag_max_1		tuple with range for random spin #1 values. if single value, s1 is kept fixed at that value
		spin_mag_max_2		tuple with range for random spin #1 values. if single value, s2 is kept fixed at that value
		t_step				time step to generate the wave with
		approximant			string for the approximant model to be used (in lal convention; to be used only if lal ought to be used)
		alpha				distorsion factor for time grid. (In range (0,1], when it's close to 0, more grid points are around merger)
		path_TEOBResumS		path to a local installation of TEOBResumS with routine 'EOBRun_module' (if given, it overwrites the aprroximant entry)
	Output:
		if filename is given
			None
		if filename is not given
			theta_vector (N_data,3)		vector holding ordered set of parameters used to generate amp_dataset and ph_dataset
			amp_dataset (N_data,N_grid)	dataset with amplitudes
			ph_dataset (N_data,N_grid)	dataset with phases
			times (N_grid,)				vector holding times at which waves are evaluated (t=0 is the time of maximum amplitude)
	"""
	d=1.
	inclination = 0.#np.pi/2.

	if path_TEOBResumS is not None:
		approximant = "TEOBResumS"

	if approximant == "TEOBResumS":
		#see https://bitbucket.org/eob_ihes/teobresums/src/development/ for the implementation of TEOBResumS
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
			raise RuntimeError("Impossible to load lalsimulation: try pip install lalsuite")
		LALpars = lal.CreateDict()
		approx = lalsim.SimInspiralGetApproximantFromString(approximant)
		

		#checking if N_grid is fine
	if not isinstance(N_grid, int):
		raise TypeError("N_grid is "+str(type(N_grid))+"! Expected to be a int.")

	if isinstance(m2_range, tuple):
		D_theta = 4 #m2 must be included as a feature
	else:
		D_theta = 3

		#creating time_grid
	t_end = 5.2e-4 #estimated maximum time for ringdown: WF will be killed after that time
	#alpha = 0.35 #exponent for "time distorsion"
	time_grid = np.linspace(-np.power(np.abs(t_coal), alpha), np.power(t_end, alpha), N_grid)
	time_grid = np.multiply( np.sign(time_grid) , np.power(np.abs(time_grid), 1./alpha))

		#adding 0 to time grid
	index_0 = np.argmin(np.abs(time_grid))
	time_grid[index_0] = 0. #0 is alway set in the grid

		#setting t_coal_freq for generating a waves
	if np.abs(t_coal) < 0.05:
		t_coal_freq = 0.05
	else:
		t_coal_freq = np.abs(t_coal)


	if filename is not None: #doing header if file is empty - nothing otherwise
		if not os.path.isfile(filename): #file doesn't exist: must be created with proper header
			filebuff = open(filename,'w')
			print("New file ", filename, " created")
			freq_header = np.concatenate((np.zeros((3,)), time_grid, time_grid) )
			freq_header = np.reshape(freq_header, (1,len(freq_header)))
			np.savetxt(filebuff, freq_header, header = "row: theta "+str(D_theta)+" | amp (1,"+str(N_grid)+")| ph (1,"+str(N_grid)+")\nN_grid = "+str(N_grid)+" | t_coal ="+str(t_coal)+" | t_step ="+str(t_step)+" | q_range = "+str(q_range)+" | m2_range = "+str(m2_range)+" | s1_range = "+str(s1_range)+" | s2_range = "+str(s2_range), newline = '\n')
		else:
			filebuff = open(filename,'a')

	if filename is None:
		amp_dataset = np.zeros((N_data,N_grid)) #allocating storage for returning data
		ph_dataset = np.zeros((N_data,N_grid))
		theta_vector = np.zeros((N_data,D_theta))

	for i in range(N_data): #loop on data to be created
		if i%50 == 0 and i != 0:
		#if i%1 == 0 and i != 0: #debug
			print("Generated WF ", i)

			#setting value for data
		if isinstance(m2_range, tuple):
			m2 = np.random.uniform(m2_range[0],m2_range[1])
		elif m2_range is not None:
			m2 = float(m2_range)
		if isinstance(q_range, tuple):
			q = np.random.uniform(q_range[0],q_range[1])
		else:
			q = float(q_range)
		if isinstance(s1_range, tuple):
			spin1z = np.random.uniform(s1_range[0],s1_range[1])
		else:
			spin1z = float(s1_range)
		if isinstance(s2_range, tuple):
			spin2z = np.random.uniform(s2_range[0],s2_range[1])
		else:
			spin2z = float(s2_range)

		if m2_range is None:
			m2 = 20. / (1+q)
			m1 = q * m2
		else:
			m1 = q* m2

			#computing f_min
		f_min = .9* ((151*(t_coal_freq)**(-3./8.) * (((1+q)**2)/q)**(3./8.))/(m1+m2))
		 #in () there is the right scaling formula for frequency in order to get always the right reduced time
		 #this should be multiplied by a prefactor (~1) for dealing with some small variation due to spins

			#getting the wave
		if approximant != "TEOBResumS": #using lal to create WFs
			hptilde, hctilde = lalsim.SimInspiralChooseTDWaveform( #where is its definition and documentation????
				m1*lalsim.lal.MSUN_SI, #m1
				m2*lalsim.lal.MSUN_SI, #m2
				0., 0., spin1z, #spin vector 1
				0., 0., spin2z, #spin vector 2
				d*1e6*lalsim.lal.PC_SI, #distance to source
				inclination, #inclination
				0., #phi ref
				0., #longAscNodes
				0., #eccentricity
				0., #meanPerAno
				t_step, # time incremental step
				f_min, # lowest value of time
				f_min, #some reference value of time (??)
				lal.CreateDict(), #some lal dictionary
				approx #approx method for the model
			)
			#print(f_min, t_step)#debug
			#print(m1,m2, spin1z,spin2z) #debug

			h_p, h_c = np.array(hptilde.data.data), np.array(hctilde.data.data) #complex waveform
			time_full = np.linspace(0.0, hptilde.data.length*t_step, hptilde.data.length) #time grid at which wave is computed

		if approximant == "TEOBResumS": #using TEOBResumS
			pars = {
				'M'                  : m1+m2,
				'q'                  : m1/m2,
				'Lambda1'            : 0.,
				'Lambda2'            : 0.,     
				'chi1'               : spin1z,
				'chi2'               : spin2z,
				'domain'             : 0,      # TD
				'arg_out'            : 0,      # Output hlm/hflm. Default = 0
				'use_mode_lm'        : [1],      # List of modes to use/output through EOBRunPy
				'srate_interp'       : 1./t_step,  # srate at which to interpolate. Default = 4096.
				'use_geometric_units': 0,      # Output quantities in geometric units. Default = 1
				'initial_frequency'  : f_min,   # in Hz if use_geometric_units = 0, else in geometric units
				'interp_uniform_grid': 2,      # Interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
				'distance': d,
				'inclination':inclination,
				'nqc_coefs_flx': 2 # {"none", "nrfit_nospin20160209", "fit_spin_202002", "fromfile"}
			}
			time_full, h_p, h_c = EOBRun_module.EOBRunPy(pars)

		if isinstance(m2_range, tuple):
			temp_theta = [m1, m2, spin1z, spin2z]		
		else:
			temp_theta = [m1/m2, spin1z, spin2z]

		temp_amp = np.sqrt(np.square(h_p)+np.square(h_c))
		temp_ph = np.unwrap(np.arctan2(h_c,h_p))

		time_full = (time_full - time_full[np.argmax(temp_amp)])/(m1+m2) #grid is scaled to standard grid
			#setting waves to the chosen std grid
		temp_amp = np.interp(time_grid, time_full, temp_amp)
		temp_ph = np.interp(time_grid, time_full, temp_ph)

			#here you need to decide what is better
		#temp_ph = temp_ph - temp_ph[0] #all phases are shifted by a constant to make every wave start with 0 phase
		id0 = np.where(time_grid == 0)[0]
		temp_ph = temp_ph - temp_ph[id0] #all phases are shifted by a constant to make every wave start with 0 phase at t=0 (i.e. at maximum amplitude)

			#removing spourious gaps (if present) (do I need it??)
		(index,) = np.where(temp_amp/np.max(temp_amp) < 1e-5) #there should be a way to choose the right threshold...
		if len(index) >0:
			print("Wave killed")
			temp_amp[index] = temp_amp[index[0]-1]
			temp_ph[index] = temp_ph[index[0]-1]

		if filename is None:
			amp_dataset[i,:] = temp_amp  #putting waveform in the dataset to return
			ph_dataset[i,:] =  temp_ph  #phase
			theta_vector[i,:] = temp_theta

		if filename is not None: #saving to file
			to_save = np.concatenate((temp_theta, temp_amp, temp_ph))
			to_save = np.reshape(to_save, (1,len(to_save)))
			np.savetxt(filebuff, to_save)

	if filename is None:
		return theta_vector, amp_dataset.real, ph_dataset.real, time_grid
	else:
		filebuff.close()
		return None

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

def generate_waveform_TEOBResumS(m1,m2, s1=0.,s2 = 0.,d=1., iota = 0., t_coal = 0.4, t_step = 5e-5, f_min = None, t_min = None, verbose = False, path_TEOBResumS = None):
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
		h_p (D,)	plus polarization of the wave
		h_c (D,)	cross polarization of the wave
	"""
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

	pars = {'M'                  : m1+m2,
			'q'                  : m1/m2,
			'Lambda1'            : 0.,
			'Lambda2'            : 0.,     
			'chi1'               : s1,
			'chi2'               : s2,
			'domain'             : 0,      # TD
			'arg_out'            : 0,      # Output hlm/hflm. Default = 0
			'use_mode_lm'        : [1],      # List of modes to use/output through EOBRunPy
			'srate_interp'       : 1./t_step,  # srate at which to interpolate. Default = 4096.
			'use_geometric_units': 0,      # Output quantities in geometric units. Default = 1
			'initial_frequency'  : f_min,   # in Hz if use_geometric_units = 0, else in geometric units
			'interp_uniform_grid': 2,      # Interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
			'distance': d,
			'inclination': iota,
			'nqc_coefs_flx': 2 # {"none", "nrfit_nospin20160209", "fit_spin_202002", "fromfile"}
		}
	times, h_p, h_c = EOBRun_module.EOBRunPy(pars)
	amp = np.sqrt(h_p**2+h_c**2)

	t_m =  times[np.argmax(amp)]
	times = times - t_m

	if t_min is not None:
		arg = np.argmin(np.abs(times+t_min))
	else:
		arg=0

	return times[arg:], h_p[arg:], h_c[arg:]




def create_dataset_FD(N_data, N_grid = None, filename = None, q_range = (1.,5.), m2_range = 20., s1_range = (-0.8,0.8), s2_range = (-0.8,0.8), log_space = True, f_high = 2000, f_step = 1e-2, f_max = None, f_min =None, approximant = "IMRPhenomPv2"):
	"""
	Create a dataset for training a ML model to fit GW waveforms in frequency domain.
	The dataset consists in 3 parameters theta=(q, spin1z, spin2z) associated to the waveform computed in frequency domain for a grid of N_grid points in the range given by the user.
	More specifically, data are stored in 3 vectors:
		theta_vector	vector holding source parameters q, spin1, spin2
		amp_vector		vector holding amplitudes for each source evaluated at some N_grid equally spaced points
		ph_vector		vector holding phase for each source evaluated at some N_grid equally spaced points
	This routine add N_data data to filename if one is specified (if file is not empty it must contain data with the same N_grid); otherwise the datasets are returned as np vectors. 
	All the waves are evaluated at a constant distance of 1Mpc. Values of q and m2 are drawn randomly in the range given by the user: it holds m1 = q *m2 M_sun.
	The waveforms are computed from f_low = 15 to f_high with a step f_step and then evaluated at some N_grid grid points equally spaced in range [f_min, f_max]
	Dataset can be loaded with load_dataset
	Input:
		N_data			size of dataset
		N_grid			number of points to be sampled in the grid (if None every point generated is saved)
		filename		name of the file to save dataset in (If is None, nothing is saved on a file)
		q_range			tuple with range for random q values. if single value, q is kept fixed at that value
		m2_range		tuple with range for random m2 values. if single value, m2 is kept fixed at that value
		spin_mag_max_1	tuple with range for random spin #1 values. if single value, s1 is kept fixed at that value
		spin_mag_max_2	tuple with range for random spin #1 values. if single value, s2 is kept fixed at that value
		log_space		whether grid should be computed in logspace
		f_high			highest frequency to compute
		f_step			step considered for computation of waveforms
		f_max			maximum frequency returned to the user (if None is the same as f_max)
		f_min			minimum frequency returned to the user (if None is the same as f_low = 15)
		approximant	string for the approximant model to be used (in lal convention)
	Output:
		if filename is given
			None
		if filename is not given
			theta_vector (N_data,3)		vector holding ordered set of parameters used to generate amp_dataset and ph_dataset
			amp_dataset (N_data,N_grid)	dataset with amplitudes
			ph_dataset (N_data,N_grid)	dataset with phases
			frequencies (N_grid,)		vector holding frequencies at which waves are evaluated
	"""
	if f_max is None:
		f_max = f_high
	if f_min is None:
		f_min = 1.
	f_low = f_min
	K = int((f_max-f_min)/f_step) #number of data points to be taken from the returned vector
	if N_grid is None:
		N_grid = K
	full_freq = np.arange(f_low, f_max, f_step) #full frequency vector as returned by lal
	d=1.
	LALpars = lal.CreateDict()
	approx = lalsim.SimInspiralGetApproximantFromString(approximant)

		#allocating storage for temp vectors to save a single WF
	temp_amp = np.zeros((N_grid,))
	temp_ph = np.zeros((N_grid,))
	temp_theta = np.zeros((3,))

		#setting frequencies to be returned to user
	if log_space:
		frequencies = np.logspace(np.log10(f_min), np.log10(f_max), N_grid)
	else:
		frequencies = np.linspace(f_min, f_max, N_grid)
		#freq_to_choose = np.arange(0, K, K/N_grid).astype(int) #choosing proper indices s.t. dataset holds N_grid points
		#frequencies = full_freq[freq_to_choose] #setting only frequencies to be chosen

	if filename is not None: #doing header if file is empty - nothing otherwise
		if not os.path.isfile(filename): #file doesn't exist: must be created with proper header
			filebuff = open(filename,'w')
			print("New file ", filename, " created")
			freq_header = np.concatenate((np.zeros((3,)), frequencies, frequencies) )
			freq_header = np.reshape(freq_header, (1,len(freq_header)))
			np.savetxt(filebuff, freq_header, header = "row: theta 3 | amp "+str(frequencies.shape[0])+"| ph "+str(frequencies.shape[0])+"\nN_grid = "+str(N_grid)+" | f_step ="+str(f_step)+" | q_range = "+str(q_range)+" | s1_range = "+str(s1_range)+" | s2_range = "+str(s2_range), newline = '\n')
		else:
			filebuff = open(filename,'a')

	if filename is None:
		amp_dataset = np.zeros((N_data,N_grid)) #allocating storage for returning data
		ph_dataset = np.zeros((N_data,N_grid))
		theta_vector = np.zeros((N_data,3))

	for i in range(N_data): #loop on data to be created
		if i%100 == 0 and i != 0:
			print("Generated WF ", i)

			#setting value for data
		if isinstance(m2_range, tuple):
			m2 = np.random.uniform(m2_range[0],m2_range[1])
		else:
			m2 = m2_range
		if isinstance(q_range, tuple):
			m1 = np.random.uniform(q_range[0],q_range[1]) * m2
		else:
			m1 = q_range * m2
		if isinstance(s1_range, tuple):
			spin1z = np.random.uniform(s1_range[0],s1_range[1])
		else:
			spin1z = s1_range
		if isinstance(s2_range, tuple):
			spin2z = np.random.uniform(s2_range[0],s2_range[1])
		else:
			spin2z = s2_range

			#debug!!!
		if m1/m2 >4.6 and (np.abs(spin1z) >0.8 or np.abs(spin2z) > 0.8):
			continue

			#getting the wave
		hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform( #where is its definition and documentation????
			m1*lalsim.lal.MSUN_SI, #m1
			m2*lalsim.lal.MSUN_SI, #m2
			0., 0., spin1z, #spin vector 1
			0., 0., spin2z, #spin vector 2
			d*1e6*lalsim.lal.PC_SI, #distance to source
			0., #inclination
			0., #phi ref
			0., #longAscNodes (for precession)
			0., #eccentricity
			0., #meanPerAno (for precession)
			f_step, # frequency incremental step
			f_low, # lowest value of frequency
			f_high, # highest value of frequency
			f_low, #some reference value of frequency (??)
			LALpars, #some lal dictionary
			approx #approx method for the model
			)
		h = np.array(hptilde.data.data)+1j*np.array(hctilde.data.data) #complex waveform
		temp_theta = [m1/m2, spin1z, spin2z]
		temp_amp = (np.abs(h)[int(f_min/f_step):int(f_max/f_step)].real)
		temp_ph = (np.unwrap(np.angle(h))[int(f_min/f_step):int(f_max/f_step)].real)

			#bringing waves on the chosen grid
		temp_amp = np.interp(frequencies, full_freq, temp_amp)
		temp_ph = np.interp(frequencies, full_freq, temp_ph)
		#temp_ph = temp_ph[freq_to_choose]; temp_amp = temp_amp[freq_to_choose] #old version of code

		temp_ph = temp_ph - temp_ph[0] #all frequencies are shifted by a constant to make the wave start at zero phase!!!! IMPORTANT

			#removing spourious gaps (if present)
		(index,) = np.where(temp_amp/temp_amp[0] < 5e-3) #there should be a way to choose right threshold...
		if len(index) >0:
			temp_ph[index] = temp_ph[index[0]-1]

		if filename is None:
			amp_dataset[i,:] = temp_amp  #putting waveform in the dataset to return
			ph_dataset[i,:] =  temp_ph  #phase
			theta_vector[i,:] = temp_theta

		if filename is not None: #saving to file
			to_save = np.concatenate((temp_theta,temp_amp, temp_ph))
			to_save = np.reshape(to_save, (1,len(to_save)))
			np.savetxt(filebuff, to_save)

	if filename is None:
		return theta_vector, amp_dataset.real, ph_dataset.real, frequencies
	else:
		filebuff.close()
		return None

def save_dataset(filename, theta_vector, dataset1, dataset, x_grid):
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
	np.savetxt(filename, to_save, header = "row: theta 3 | amp "+str(amp_dataset.shape[1])+"| ph "+str(ph_dataset.shape[1])+"\nN_grid = "+str(x_grid.shape[0])+" | f_step ="+str(x_step)+" | q_max = "+str(q_max)+" | spin_mag_max = "+str(spin_mag_max), newline = '\n')
	return


def load_dataset(filename, N_data=None, N_grid = None, shuffle = False):
	"""
	Load a GW dataset from file. The file should be suitable for np arrays and have the following structure:
		theta 3 | amplitudes K | phases K
	The first row hold the frequncy vector.
	It can shuffle the data if required.
	Input:
		filename	input filename
		N_data		number of data to extract (only if data in file are more than N_data) (if None N_data = N)
		N_grid		number of grid points to evaluate the waves in (Only if N_grid < N_grid_dataset)
		shuffle		whether to shuffle data
	Outuput:
		theta_vector (N_data,3)	vector holding ordered set of parameters used to generate amp_dataset and ph_dataset
		amp_dataset (N_data,K)	dataset with amplitudes and wave parameters K = (f_high-30)/(f_step*N_grid)
		ph_dataset (N_data,K)	dataset with phases and wave parameters K = (f_high-30)/(f_step*N_grid)
		x_grid (K,)				vector holding x_grid at which waves are evaluated (can be frequency or time grid)
	"""
	if N_data is not None:
		N_data += 1
	data = np.loadtxt(filename, max_rows = N_data)
	N = data.shape[0]
	K = int((data.shape[1]-3)/2)
	x_grid = data[0,3:3+K] #saving x_grid

	data = data[1:,:] #removing x_grid from full data
	if shuffle: 	#shuffling if required
		np.random.shuffle(data)

	theta_vector = data[:,0:3]
	amp_dataset = data[:,3:3+K]
	ph_dataset = data[:,3+K:]

	if N_grid is not None:
		if ph_dataset.shape[1] < N_grid:
			print("Not enough grid points ("+str(ph_dataset.shape[1])+") for the required N_grid value ("+str(N_grid)+").\nMaximum number of grid point is taken (but less than N_grid)")
			N_grid = ph_dataset.shape[1]
		indices = np.arange(0, ph_dataset.shape[1], int(ph_dataset.shape[1]/N_grid)).astype(int)
		x_grid = x_grid[indices]
		ph_dataset = ph_dataset[:,indices]
		amp_dataset = amp_dataset[:,indices]

	return theta_vector, amp_dataset, ph_dataset, x_grid

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

