import numpy
from ctypes import *
from numpy.ctypeslib import ndpointer

lib = cdll.LoadLibrary('./npyinterp.so')
lib.interpolate_integrate.argtypes = [
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int]

def interp(left, right, x, y):
	"""
	Integration and interpolation
	
	Data with coordinates x and values at those coordinates y is 
	piecewise linearly interpolated and integrated within 
	bins. The bin borders are defined by left and right.
	
	:param: left:  Lower bin border (numpy array)
	:param: right: Upper bin border (numpy array, same length as left)
	:param: x:     Coordinates where data values are defined (numpy array)
	:param: y:     Data values at those coordinates (numpy array, same 
	               length as x)
	:return: numpy array (of the same length as left/right) containing the
	   integration within the bins.	
	
	left and right each have to be monotonically increasing (and right > left).
	Numpy arrays have to be float64 and contiguous (i.e. fancy indexing can not be used
	directly, a copy is necessary.).
	
	This function is most useful for Sherpa models.
	"""
	## sherpa-2> %timeit calc_kcorr(z=3, obslo=0.2,obshi=2)
	## 1000 loops, best of 3: 1.94 ms per loop
	## (2577 times faster than atable)
	#print 'using interpolation library', len(left), len(right), len(x), len(y)
	z = numpy.zeros_like(left) - 1
	r = lib.interpolate_integrate(left, right, z, len(left), x, y, len(x))
	if r != 0:
		raise Exception("Interpolation failed")
	return z


