import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes


lib = ctypes.cdll.LoadLibrary('./interp.so') #loading library
lib.interp_N.argtypes = [
	ctypes.c_int,
	ctypes.c_int,
	ctypes.c_int,
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
	]

lib.interp.argtypes = [
	ctypes.c_int,
	ctypes.c_int,
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
	ctypes.c_double,
	ctypes.c_double
	]

def interp_N(x, xp, yp, left = None, right = None):
	"""
	Interpolation with the same interface as np.interp, but runned in C++. It is around a factor or 2 faster.
	It also add the chance for a multidimensional function.
	Input:
		x (D',)				new grid
		xp (D,)				old grid
		yp (N,D)/(D,)		function(s) to interpolate (evaluated at xp)
		left (N,)/()		Value to return where x<xp[0] (if None, yp[:,0] is returned)
		right (N,)/()		Value to return where x>xp[0] (if None, yp[:,-1] is returned)
	Output:
		y (N,D')/(D',)		interpolated function at x
	"""
	if yp.ndim ==1:
		to_reshape = True
		yp = yp[None,:]
	else:
		to_reshape = False
	if left is None:
		left = np.array(yp[:,0])
	if not isinstance(left, np.ndarray):
		left = np.ones((yp.shape[0],))*left
	if right is None:
		right = np.array(yp[:,-1])
	if not isinstance(right, np.ndarray):
		right = np.ones((yp.shape[0],))*right

	lib.interp_N.restype = ndpointer(dtype=ctypes.c_double, shape=(yp.shape[0],x.shape[0]))
	y = lib.interp_N(len(x), len(xp), yp.shape[0], x, xp, yp, left, right)
	if not to_reshape:
		return y
	else:
		return y[0,:]


def interp(x, xp, yp, left = None, right = None):
	"""
	Interpolation with the same interface as np.interp, but runned in C++. It is around a factor or 2 faster.
	Input:
		x (D',)		new grid
		xp (D,)		old grid
		yp (D,)		function(s) to interpolate (evaluated at xp)
		left 		Value to return where x<xp[0] (if None, yp[:,0] is returned)
		right 		Value to return where x>xp[0] (if None, yp[:,-1] is returned)
	Output:
		y (D',)		interpolated function at x
	"""
	if left is None:
		left = yp[0]
	if right is None:
		right = yp[-1]

	lib.interp.restype = ndpointer(dtype=ctypes.c_double, shape=x.shape)
	y = lib.interp(len(x), len(xp), x, xp, yp, left, right)
	return y




	
