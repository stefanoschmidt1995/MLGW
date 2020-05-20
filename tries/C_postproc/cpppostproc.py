import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes

lib = ctypes.cdll.LoadLibrary('./postproc.so') #loading library	
lib.post_process.argtypes = [
	ctypes.c_int, #N_data
	ctypes.c_int, #D_std
	ctypes.c_int, #D_us
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), #t_std
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), #t_us
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), #amp
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), #ph
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), #m_tot
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), #d_L
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), #iota
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')  #phi_ref
	]


def post_process(t_std, t_us, amp, ph, m_tot, d_L, iota, phi_ref):
	"""
	Interpolation + post processing of amplitude and phase. Interpolates to t_us from WFs evaluated at t_std. Included dependence on geometrical parameters (luminosity distance, inclination and reference phase).
	Input:
		t_std (D,)		standard grid (reduced time)
		t_us (D',)		user chosen grid
		amp (N,D)/(D,)	amplitude (evaluated at t_std)
		ph (N,D)/(D,)	amplitude (evaluated at t_std)
		m_tot (N,)		total mass of the WFs
		d_L (N,)		luminosity distance
		iota (N,)		inclination
		phi_ref (N,)	reference phase
	Output:
		h_p (N,D')/(D',)	plus polarization
		h_c (N,D')/(D',)	cross polarization
	"""
	if amp.ndim ==1:
		to_reshape = True
		amp = amp[None,:]
		ph = ph[None,:]
		assert amp.shape == ph.shape
	else:
		to_reshape = False

	N = amp.shape[0]
	D_std = len(t_std)
	D_us = len(t_us)

	#print("ciO",amp.flatten(), amp)

	lib.post_process.restype = ndpointer(dtype=ctypes.c_double, shape=(N*D_us*2))
	h = lib.post_process(N, D_std, D_us, t_std, t_us, amp.flatten(), ph.flatten(), np.array(m_tot), np.array(d_L), np.array(iota), np.array(phi_ref))
	h = np.reshape(h, (N,D_us,2), order ='C')
	#print(h[1,:,1])
	if not to_reshape:
		return h[:,:,0],h[:,:,1]
	else:
		return h[0,:,0],h[0,:,1]





	
