import numpy as np
import matplotlib.pyplot as plt
import time
import cpppostproc #importing the package
import mlgw.GW_generator as gen

a = gen.GW_generator()

theta = np.array([[25,5,0.3,-0.85, 2.14, 2.32, 1.30],[25,5,-0.3,-0.85, 2.14, 2.32, 1.30]])#, [15,1,-0.3,-0.5, 1, 0, 0], [18,5,0.13,-0.15, 10, 2.2, 1.21],[11,9,-0.3,-0.75, .63, 0.1, 1.34]])
std_theta = np.column_stack([theta[:,0]/theta[:,1], theta[:,2],theta[:,3]])
#print(std_theta)

t_us = np.linspace(-50,0.1,100000)

start = time.process_time_ns()/1e6 #ms
	#mlgw
h_p, h_c = a.get_WF(theta, t_us)#, red_grid = True)

middle = time.process_time_ns()/1e6 #ms
	# C++ mlgw
amp, ph = a.get_raw_WF(std_theta)
h_p_rec, h_c_rec = cpppostproc.post_process(a.get_time_grid(), t_us, amp, ph, theta[:,0]+theta[:,1],theta[:,4], theta[:,5], theta[:,6])

end = time.process_time_ns()/1e6 #ms

print("times: speed up, mlgw, C", (middle-start)/(end-middle), (middle-start), (end-middle))

n_WF = 1

print("hp: ", np.allclose(1e22*h_p, 1e22*h_p_rec))
print("hc: ", np.allclose(1e22*h_c, 1e22*h_c_rec))

plt.title(r"$h_+$")
#plt.plot(t_us[:],np.sqrt(h_p[0,:]**2+h_c[0,:]**2), label = 'true')
plt.plot(t_us[:],h_p[n_WF,:], '-', ms = 1, label = 'true')
plt.plot(t_us[:],h_p_rec[n_WF,:], '--', ms = 1, label = 'rec')
plt.legend()
plt.figure()
plt.title(r"$h_\times$")
#plt.plot(t_us[:],np.unwrap(np.arctan2(h_c[0,:],h_p[0,:])), label = 'true')
plt.plot(t_us[:],h_c[n_WF,:], '-', ms = 1, label = 'true')
plt.plot(t_us[:],h_c_rec[n_WF,:], '--', ms = 1, label = 'rec')
#plt.plot(t_us[:],np.arctan2(h_c[0,:],h_p[0,:])-h_c_rec[0,:], label = 'diff')
plt.legend()
plt.show()

print(std_theta.shape)




