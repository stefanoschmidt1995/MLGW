import monointerp
import numpy as np
import matplotlib.pyplot as plt
import time

f = lambda x: np.exp(x/100.)
# data for interpolation
#x = np.array(range(100), dtype=np.float64)*1.
#y = np.array(range(500000), dtype=np.float64)*1.
x = np.linspace(0,100,100)
y = np.linspace(0,100,500000)

# lower and upper bin borders to integrate within

#r = np.array(range(10000 - 10), dtype=np.float64)*1. + 3.1
#q = np.zeros(r.shape) #np.array(range(10000 - 10), dtype=np.float64)*1. + 6.121
f_0 = f(x)

def run():
	# this is how to call the function, simple!
	return monointerp.interp(f_0, np.zeros(f_0.shape), x, y)

def run2():
	# for verification, we compare to np.interp
	return np.interp(y,x,f_0)


def test():
	start_time = time.process_time_ns()/1e6 #ms
	z = run()
	middle_time = time.process_time_ns()/1e6 #ms	
	z_check = run2()
	end_time = time.process_time_ns()/1e6 #ms
	print("monointerp vs np", middle_time-start_time, end_time-middle_time)
#	plt.plot(y, z, label = "monointerp")
	plt.plot(y, z_check, label = "numpy")
	plt.legend()
	plt.show()
	
	assert np.allclose(z, z_check)
	

if __name__ == '__main__':
	test()

