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
#"""
create_dataset_alpha_beta(N_angles = 20, filename="temp.dat", N_grid = 1500, tau_min = 20., q_range= (1.1,10.), smooth_oscillation = True, verbose = True)

params, alpha, beta_avg, beta_residual, times = load_dataset("temp.dat", N_data = None, n_params = 6, N_entries = 3)

plt.figure()
plt.title("avg")
plt.plot(times,beta_avg.T)
plt.plot(times,(beta_avg+beta_residual).T)
plt.figure()
plt.title("residual")
plt.plot(times,beta_residual.T)
plt.figure()
plt.title("alpha")
plt.plot(times,alpha.T)

plt.show()
#"""
#quit()
params, alpha, beta, times = load_dataset("grad_angles.dat", N_data = None, n_params = 6)
print("Loaded {} data".format(params.shape[0]))
N = 10

	#Averaging the function
	#doing something dirty, but maybe useful

f_min, f_max = compute_spline_peaks(times, beta)
f_mean = get_spline_mean(times, beta)

fig, ax = plt.subplots(2,1)
#np.random.seed(21)
choice = np.array([2, 26, 27, 49, 81,81, 114,121,123,130, 138])
choice = np.random.choice(range(beta.shape[0]), size =(N,), replace = False)
for i in choice:

	ax[0].plot(times, beta[i,:])
	peaks, props = scipy.signal.find_peaks(beta[i,:])
	m_peaks, props = scipy.signal.find_peaks(-beta[i,:])
	peaks = np.append(peaks, m_peaks)
	ax[0].plot(times[peaks], beta[i,peaks], 'o', ms = 3)#(f_grad_max[i](times)+f_grad_min[i](times))/2.)
	ax[0].plot(times, (f_max[i](times)+f_min[i](times))/2.)
	ax[1].plot(times, (f_max[i](times)+f_min[i](times))/2.)
	
	assert np.allclose((f_max[i](times)+f_min[i](times))/2., f_mean(times)[i,:])

	#plotting beta-avg to see if the problem is simpler...
fig, ax = plt.subplots(2,1)
for i in choice:
	avg = (f_max[i](times)+f_min[i](times))/2.
	m_list, M_list = compute_spline_peaks(times, (beta[i,:]-avg)[None,:])
	amp = lambda t: (M_list[0](t) - m_list[0](t))/2.
	cutoff = 950
	
	ax[0].plot(times, amp(times),'-.')#,'o', ms = 3)
	ax[0].plot(times, beta[i,:]-avg)
	if np.all((beta[i,:]-avg) == 0 ):
		ax[1].plot(times[:cutoff], np.zeros(times[:cutoff].shape))
	else:
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


