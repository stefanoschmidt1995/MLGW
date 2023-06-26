#Here we are comparing the two methods for computing the 22 mode.
#The standard one: a loop over all modes and a general method for computing the spherical harmonics
#The optimized one: no loop over all modes and a fast expression for the spherical harmonics
#You can really see the difference between the two approaches! Especially when a single WF is considered: you might gain a factor of 3


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../../mlgw_v2')
import time
import GW_generator

#getting random theta
m1_range = (10.,100.)
m2_range = (10.,100.)
s1_range = (-0.8,0.8)
s2_range = (-0.8,0.8)
d_range = (.5,10.)
i_range = (0,np.pi) 
phi_0_range = (0,3.1415)

low_list = [m1_range[0],m2_range[0], s1_range[0], s2_range[0], d_range[0], i_range[0], phi_0_range[0]]
high_list = [m1_range[1],m2_range[1], s1_range[1], s2_range[1], d_range[1], i_range[1], phi_0_range[1]]

N_waves = 1

theta = np.random.uniform(low = low_list, high = high_list, size = (N_waves, 7))

times = np.linspace(-8,0.01,100000)

gen = GW_generator.GW_generator()

start = time.time()
h_p, h_c = gen.get_WF(theta,times, (2,2))
middle = time.time()

h_p_slow, h_c_slow = gen.get_WF(theta,times, [(2,2)])
end = time.time()

print("Time profile: opt/slow", middle-start, end-middle)

print("The two WFs are the same? :", np.allclose(h_p_slow,h_p,atol=0) and np.allclose(h_c_slow,h_c,atol=0))









