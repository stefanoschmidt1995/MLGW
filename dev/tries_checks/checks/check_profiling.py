import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as lalsim
import lal
import sys
sys.path.insert(1, '../mlgw_v1') #folder in which every relevant routine is saved

from GW_generator import * 	#routines for dealing with datasets
from GW_helper import * 	#routines for dealing with datasets

import cProfile
import pstats
try:
	from line_profiler import LineProfiler

	def do_profile(follow=[]):
		def inner(func):
			def profiled_func(*args, **kwargs):
				try:
					profiler = LineProfiler()
					profiler.add_function(func)
					for f in follow:
						profiler.add_function(f)
					profiler.enable_by_count()
					return func(*args, **kwargs)
				finally:
					profiler.print_stats()
			return profiled_func
		return inner
except:
	pass
generate_stats = True

	#defining generator and theta
generator = GW_generator("../mlgw_v1/TD_model_TEOBResumS")

m1_range = (5.,50.)
m2_range = (5.,50.)
s1_range = (-0.8,0.8)
s2_range = (-0.8,0.8)
d_range = (.5,10.)
i_range = (0,2*np.pi)
phi_0_range = (0,np.pi)

low_list = [m1_range[0],m2_range[0], s1_range[0], s2_range[0], d_range[0], i_range[0], phi_0_range[0]]
high_list = [m1_range[1],m2_range[1], s1_range[1], s2_range[1], d_range[1], i_range[1], phi_0_range[1]]
times = np.linspace(-6.,0.05,1000)

def generate_waves(N_waves = 16):
	theta = np.random.uniform(low = low_list, high = high_list, size = (N_waves, 7))
		#creating theta_std
	#theta_std  = np.column_stack((theta[:,0]/theta[:,1],theta[:,2],theta[:,3])) #(N,3)
	#to_switch = np.where(theta_std[:,0] < 1.) #holds the indices of the events to swap
	#theta_std[to_switch,0] = np.power(theta_std[to_switch,0], -1)
	#theta_std[to_switch,1], theta_std[to_switch,2] = theta_std[to_switch,2], theta_std[to_switch,1]

	#for i in range(N_waves):
	h_p, h_c = generator.get_WF(theta, t_grid = times)

	#plt.plot(times,h_p[0,:])
	#plt.show()

	return h_p,h_c

#doing profiling

#plt.plot(times, h_p[0,:])
#plt.show()

if generate_stats:
	#generator.get_red_coefficients(np.array([[1.5,-0.58,0.3],[2.5,0.2,-0.13]]))
	#cProfile.run('generate_waves(100)', 'wave_stats')
	generate_waves(100)
	#quit()

p = pstats.Stats('wave_stats')
p.strip_dirs()
p.sort_stats('tottime')
#p.print_stats()














