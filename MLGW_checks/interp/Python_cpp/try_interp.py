import numpy as np
import matplotlib.pyplot as plt
import time
import cppinterp #wrapper to C++ function where the interpolation is performed

N_data = 200
x = np.linspace(0,100,20)
y = np.exp(x/100.)
x_new = np.linspace(-10,110,500000)

y = np.repeat(y[None,:],N_data, axis =0)
y_np = np.zeros((N_data, len(x_new)))
y_new = np.zeros((N_data, len(x_new)))

start_time = time.process_time_ns()/1e6 #ms
for i in range(N_data):
	y_np[i,:] = np.interp(x_new, x, y[i,:])
middle_time = time.process_time_ns()/1e6
for i in range(N_data):
	y_new[i,:] = cppinterp.interp(x_new, x, y[i,:])
second_middle_time = time.process_time_ns()/1e6
y_new = cppinterp.interp_N(x_new, x, y, 0,0) #fast C++ interpolation
end_time = time.process_time_ns()/1e6

print("# interpolation: ", N_data, "\nNew grid size: ", len(x_new))
print("np vs cpp (par) vs cpp: ", middle_time-start_time, end_time-second_middle_time, second_middle_time-middle_time)


plt.plot(x, y[0,:],'-',label = "True", ms =2)
plt.plot(x_new,y_new[15,:], '-.', label = "Interp", ms =2)
plt.plot(x_new,y_np[15,:], '--', label = "numpy", ms =2)
plt.legend(loc = "upper left")
plt.show()






