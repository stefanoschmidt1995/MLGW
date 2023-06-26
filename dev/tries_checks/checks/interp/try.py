import numpy as np
import matplotlib.pyplot as plt
import time

from numba import jit

@jit(nopython=True)
def interp(x, xp, yp, extrapolate_behaviour = "const"):
	ip = 0
	ip_next = 1
	i = 0
	y = np.zeros((len(x),))
	while i < len(x):
		m = (yp[ip_next]-yp[ip])/(xp[ip_next]-xp[ip])
		q = yp[ip] - m* xp[ip]
		while x[i]<xp[ip_next]:# or ip_next == -1:
			if x[i]>=xp[ip]:
				y[i] = m*x[i]+q
			else:
				if extrapolate_behaviour == "const":
					y[i] = yp[0]
				pass
			i +=1
			if i >= len(x): break
		ip +=1
		ip_next +=1
		if ip_next == len(xp):
			if extrapolate_behaviour == "const": y[i:] = yp[-1]
			break

	return y
		

x = np.linspace(0,100,100)
y = np.exp(x/100.)
x_new = np.linspace(0,100,500000)
#indices = np.random.permutation(x.shape[0])#range(x.shape[0])
#indices_new = np.random.permutation(x_new.shape[0])
#x = x[indices]
#y = y[indices]

#x_new = x_new[indices_new]

start_time = time.process_time_ns()/1e6 #ms
y_new_np = np.interp(x_new,x,y)
middle_time = time.process_time_ns()/1e6
y_new = interp(x_new,x,y)
end_time = time.process_time_ns()/1e6

print("Number of points: ", len(x_new))
print("Interpolations agree? ",np.allclose(y_new, y_new_np))
print("np vs mine", middle_time-start_time, end_time-middle_time)

plt.plot(x,y,'o', ms =2)
plt.plot(x_new,y_new,'--')
plt.show()











