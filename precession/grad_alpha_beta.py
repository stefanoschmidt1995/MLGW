import numpy as np
import matplotlib.pyplot as plt
import scipy.signal, scipy.interpolate

from precession_helper import *

import sys
sys.path.insert(0,'../../mlgw_v2')
from GW_helper import *
from ML_routines import *

	#snippet to show that a constant f is not easy to retrieve from data
#t = np.linspace(-20,-0.1,10000)
#ph = np.cos(10.*t)
#plt.plot(t, (np.gradient(np.arccos(ph)/t, t)))
#plt.plot(t,ph)
#plt.plot(t, np.arccos(ph)/t)
#plt.plot(np.fft.rfftfreq(len(ph),np.diff(t)[0]),np.fft.rfft(ph))
#plt.show()

def compute_spline_peaks(x,y):
	max_list = []
	min_list = []
	for i in range(y.shape[0]):
		max_peaks, props = scipy.signal.find_peaks(y[i,:])
		min_peaks, props = scipy.signal.find_peaks(-y[i,:])
		#max_peaks = np.append(max_peaks,len(y[i,:])-1)
		#min_peaks = np.append(min_peaks,len(y[i,:])-1)

		if len(max_peaks) > 5 and len(min_peaks) > 5:
			max_list.append(scipy.interpolate.CubicSpline(x[max_peaks], y[i, max_peaks], extrapolate = True))
			min_list.append(scipy.interpolate.CubicSpline(x[min_peaks], y[i, min_peaks], extrapolate = True))
		else:
			print("Using all the data")
			max_list.append(scipy.interpolate.CubicSpline(x, y[i, :], extrapolate = True))
			min_list.append(scipy.interpolate.CubicSpline(x, y[i, :], extrapolate = True))
		
	return min_list, max_list
	
def compute_spline_extrema(x,y, get_spline = False):
	maxima = scipy.signal.argrelextrema(y, np.greater, axis = 1) #(N,M)
	minima = scipy.signal.argrelextrema(y, np.less, axis = 1) #(N,M')

	max_list = []
	min_list = []
	spline_list = []
	for i in range(y.shape[0]):
		ids_0_max = np.where(maxima[0]==i)[0]
		ids_0_min = np.where(minima[0]==i)[0]

		max_list.append(scipy.interpolate.CubicSpline(x[maxima[1][ids_0_max]], y[i, maxima[1][ids_0_max]], extrapolate = True))
		min_list.append(scipy.interpolate.CubicSpline(x[minima[1][ids_0_min]], y[i, minima[1][ids_0_min]], extrapolate = True))
		if get_spline:
			spline_list.append(scipy.interpolate.CubicSpline(x, y[i,:], extrapolate = True))
	if get_spline:
		return min_list, max_list, spline_list
	return min_list, max_list

#create_dataset_alpha_beta(N_angles = 30, filename="grad_angles.dat", N_grid = 1500, tau_min = 20., q_range= (1.1,10.))

params, alpha, beta, times = load_dataset("grad_angles.dat", N_data = None, n_params = 6)
N = 6

	#Averaging the function
	#doing something dirty, but maybe useful

f_min, f_max = compute_spline_peaks(times, beta)

plt.figure()
#np.random.seed(21)
choice = np.random.choice(range(beta.shape[0]), size =(N,), replace = False)
for i in choice:

	plt.plot(times, beta[i,:])
	peaks, props = scipy.signal.find_peaks(beta[i,:])
	plt.plot(times[peaks], beta[i,peaks], 'o', ms = 3)#(f_grad_max[i](times)+f_grad_min[i](times))/2.)
	plt.plot(times, (f_max[i](times)+f_min[i](times))/2.)


	#plotting beta-avg to see if the proble is simpler...
fig, ax = plt.subplots(2,1)
for i in choice:
	avg = (f_max[i](times)+f_min[i](times))/2.
	amp = compute_spline_peaks(times, (beta[i,:]-avg)[None,:])[1][0]
	cutoff = 950
	
	
	ax[0].plot(times, amp(times),'-.')#,'o', ms = 3)
	ax[0].plot(times, beta[i,:]-avg)
	ax[1].plot(times[:cutoff], ((beta[i,:]-avg)/amp(times))[:cutoff])
		#polishing cos args
	args = ((beta[i,:]-avg)/amp(times))[:cutoff]
	args[np.where(args > 1.)] = 1.
	args[np.where(args < -1.)] = -1.
	#ax[1].plot(times[:cutoff], np.abs(np.gradient(np.arccos(args), times[:cutoff])))
	#ph_approx = compute_spline_peaks(times[:cutoff], np.abs(np.gradient(np.arccos(args), times[:cutoff]))[None,:])[1][0]
	#ax[1].plot(times[:cutoff], ph_approx(times[:cutoff]))
	
	#ax[1].plot(times[:cutoff], np.arccos(args)/times[:cutoff])
	
	#ax[1].plot(times[:cutoff], np.abs(np.gradient(np.arccos(np.cos(10.*times[:cutoff]))/times[:cutoff], times[:cutoff])))
	
	#ax[1].plot(times, beta[i,:]-avg)
	#ax[1].plot(np.unwrap(np.angle(np.fft.rfft(args))))
	#ax[1].plot(np.abs(np.fft.rfft(args)))

plt.show()
quit()

	#Averaging betas with gradients
grad_beta =	np.gradient(beta,times, axis = 1)
f_min, f_max = compute_spline_extrema(times, grad_beta)

plt.figure()
for i in choice:
	#smoothen_grad = (f_max[i](grad_beta[i,:])/f_min[i](grad_beta[i,:]))/2.
	#peaks, props = scipy.signal.find_peaks(smoothen_grad, distance = 10)
	#plt.plot(times, smoothen_grad)
	peaks, props = scipy.signal.find_peaks(grad_beta[i,:])
	plt.plot(times, grad_beta[i,:])
	plt.plot(times[peaks], grad_beta[i,peaks], 'o', ms = 3)#(f_grad_max[i](times)+f_grad_min[i](times))/2.)


plt.figure()
plt.title("beta")
plt.plot(times, beta.T[:,5:])

plt.figure()
plt.title("beta grad")
plt.plot(times, np.gradient(beta,times, axis = 1).T[:,5:])
plt.show()


