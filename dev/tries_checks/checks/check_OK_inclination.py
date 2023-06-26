import mlgw.GW_generator as gen
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(1, '../mlgw_v1') #folder in which every relevant routine is saved

from GW_generator import * 	#routines for dealing with datasets
from GW_helper import * 	#routines for dealing with datasets

old_gen = gen.GW_generator()

new_gen = GW_generator("../mlgw_v1/TD_model_TEOBResumS")

m1_range = (5.,100.)
m2_range = (5.,100.)
s1_range = (-0.8,0.95)
s2_range = (-0.8,0.95)
d_range = (1,1)
i_range = (0,np.pi) 
phi_0_range = (0,2*np.pi)

low_list = [m1_range[0],m2_range[0], s1_range[0], s2_range[0], d_range[0], i_range[0], phi_0_range[0]]
high_list = [m1_range[1],m2_range[1], s1_range[1], s2_range[1], d_range[1], i_range[1], phi_0_range[1]]

theta = np.random.uniform(low = low_list, high = high_list, size = (1000, 7))

times = np.linspace(-100,0.1,10000)

start = time.process_time_ns()/1e6 #ms
	#mlgw
h_p, h_c = old_gen.get_WF(theta, times)#, out_type = "ampph")
print("ok")
middle = time.process_time_ns()/1e6 #ms

h_p_rec, h_c_rec = new_gen.get_WF(theta, times)#, out_type = "ampph")

end = time.process_time_ns()/1e6 #ms

F, phi_ref = compute_optimal_mismatch(h_p_rec+1j*h_c_rec,h_p+1j*h_c)
print("mean/max F", np.mean(F), np.max(F))

print("times: speed up, old, new", (middle-start)/(end-middle), (middle-start), (end-middle))

print("hp: ", np.allclose(1e20*h_p, 1e20*h_p_rec, 1e-7,1e-7))
print("hc: ", np.allclose(1e20*h_c, 1e20*h_c_rec,1e-7,1e-7))

plt.plot(times, h_c[0,:]) #ph diff
plt.plot(times, h_c_rec[0,:])
plt.figure()
plt.plot(times, h_p[0,:])
plt.plot(times,h_p_rec[0,:]) #ph diff
plt.show()

	


