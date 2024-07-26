import lal
import matplotlib.pyplot as plt
import numpy as np
import mlgw
import scipy.optimize
import scipy.signal

from PyEMD import EMD


def beta_approx(theta, ph, times, fref):

	#TODO: compute more seriously initial and final beta! After this is done, it's an amazing approximation, which you can already use for your precessing model (maybe?)

	ids_, = np.where(times>0)

	m1, m2 = theta[[0,1]]
	S1, S2 = theta[[2,3,4]], theta[[5,6,7]]
	S = S1+S2
	S_perp = np.linalg.norm(S[[0,1]])
	S_par = np.abs(S[2])
	
	M = m1+m2
	mu = (m1*m2)/M
	mu_tilde = mu**3/M**4
	mu_tilde /= 4.93e-6
	
	omega_orb = -0.5*np.gradient(ph[:,0], times)
	
	L = (mu_tilde/omega_orb)**(1./3.) # this is L/M**2
	L[ids_]=0

	tanbeta_ = S_perp/(L+S_par)
	
	print('S_perp, S_par = {} , {}\nratio = {}'.format(S_perp, S_par, S_perp/S_par))

	L_vect = np.column_stack([np.zeros(len(L)), np.zeros(len(L)), L])
	
	J = L_vect+S

	cosbeta_ = (S[2]+L)/np.linalg.norm(J, axis = 1) #For some reason this is worse...

	return tanbeta_, cosbeta_, L, omega_orb
	
def beta_guess(L, a, b):
	return a/(L+b)

def fit_L(L, beta, times):

	ids_, = np.where(times<0)
	L, beta = L[ids_], beta[ids_] #This is a very good idea!!!

	loss = lambda x: np.mean(np.square(beta_guess(L, *x) - beta))
	
	def loss_derivative(x):
		loss_i = beta_guess(L, *x) - beta
		grad_a = 2*loss_i/(L+x[1])
		grad_b = -2*loss_i*x[0]/(L+x[1])**2
		return np.array([np.mean(grad_a), np.mean(grad_b)])
	
	b0 = (beta[0]*L[0] - beta[-1]*L[-1])/(beta[-1]-beta[0])
	a0 = beta[0]*L[0] + b0*beta[0]
	
	res = scipy.optimize.minimize(loss, x0 = np.array([a0,b0]), jac = loss_derivative)
	a,b = res.x
	#print(a0, b0)
	#print(res)
	print(a,b)
	print(a/b)
	
	return res.x

def get_alpha(beta_guess_, L, omega_orb, M, A, B, C, h):

	M*=4.93e-6
	
	#plt.plot(np.cos(ph_res_beta_))
	#plt.plot(np.sin(ph_res_beta_))
	#plt.show()
	
	#J = L * A + B
	#Omega_p = (3+1.5/q)*J*M*omega_orb**2#/np.sin(beta_guess_)
	#Omega_p = (3+1.5/q)*M*(omega_orb * A + B)**2/np.sin(beta_guess_+C)
	
	#Omega_p = (3+1.5/q)*M*(omega_orb * A + B)**2#/np.sin(beta_guess_)
	
	Omega_p = (3+1.5/q)*M*(omega_orb * A + B)**2 #/(1+C*np.sin(beta_guess_))
	
	dt = np.mean(np.diff(times))

	return np.cumsum(Omega_p)*dt+h

def fit_alpha(alpha, beta_guess_, L, omega_orb, M, times):

	ids_, = np.where(times<-0.05)
	alpha, beta_guess_, L, omega_orb = alpha[ids_], beta_guess_[ids_], L[ids_], omega_orb[ids_]
	
	loss = lambda x: np.mean(np.square(get_alpha(beta_guess_, L, omega_orb, M, *x) - alpha))
	
	res = scipy.optimize.minimize(loss, x0 = np.array([1,1, 1e-3, alpha[0]]))
	print(res)
	
	return res.x

def get_amp_ph_beta_residuals(beta_residuals, times):
	hil = scipy.signal.hilbert(beta_residuals)
	return np.mean(np.abs(hil)[times<0]), np.unwrap(np.angle(hil))


def fit_res_alpha_trend_EMD(res_alpha, times):
	ids_, = np.where(times<-0.1)
	
	emd = EMD()
	imf = emd(res_alpha[ids_], times[ids_])

	oscillation = imf[1] #hopefully that's the one, but we are gonna need to be more robust
	trend = np.sum(np.delete(imf, [1], axis = 0), axis = 0)
	
	return trend, oscillation, ids_

def get_alpha_res(beta_guess_, L, omega_orb, M, A, B, C, h):

	M*=4.93e-6
	
	#plt.plot(np.cos(ph_res_beta_))
	#plt.plot(np.sin(ph_res_beta_))
	#plt.show()
	
	#J = L * A + B
	#Omega_p = (3+1.5/q)*J*M*omega_orb**2#/np.sin(beta_guess_)
	#Omega_p = (3+1.5/q)*M*(omega_orb * A + B)**2/np.sin(beta_guess_+C)
	
	#Omega_p = (3+1.5/q)*M*(omega_orb * A + B)**2#/np.sin(beta_guess_)
	
	Omega_p = (3+1.5/q)*M*(omega_orb * A + B)**2 /(1+C*np.sin(beta_guess_))
	
	dt = np.mean(np.diff(times))

	return np.cumsum(Omega_p)*dt+h

def fit_res_alpha_trend(res_alpha, times, beta, L, omega_orb, M):
	ids_, = np.where(times<-0.1)
	
	res_alpha, L, omega_orb, beta = res_alpha[ids_], L[ids_], omega_orb[ids_], beta[ids_]
	
	loss = lambda x: np.mean(np.square(get_alpha_res(beta, L, omega_orb, M, *x) - res_alpha))
	
	res = scipy.optimize.minimize(loss, x0 = np.array([1,1, 1, 0]))
	print('#######\n',res)
	
	return res.x
	
gen = mlgw.GW_generator() #creating an istance of the generator (using default model)
q = 7.05
theta = np.array([15*q/(1+q), 15/(1+q), 0.9, 0.3, -0.2, -0.1, 0.6, -0.3]) #q = 2
theta = np.array([15*q/(1+q), 15/(1+q), -0.4, 0.3, -0.2, -0.1, 0.6, 0.3])
theta_NP = theta[[0,1,4,7]]
times = np.linspace(-30, 0.02, 100000) 
modes = [(2,2)]
amp, ph = gen.get_modes(theta_NP, times, modes) #returns amplitude and phase of the wave

alpha, beta, gamma = gen.get_alpha_beta_gamma(theta, times, 5, 5)

tanbeta_, cosbeta_, L, omega_orb = beta_approx(theta, ph, times, fref = 5)

#beta = np.tan(beta)

# TAKE-AWAY:
#	The ansatz a/(L+b) is a good ansatx for beta itself, not tan(beta) as you would expect...
# TODO: Try to make an expansion in L, so maybe you can add another term in L^2
# TODO: Try to fit the residual with a time dependent NN: you're gonna rock everything... 
# TODO: Can you make an approximation of the frequency of the residuals? They should happen at the Omega_p scale sort of...
# TODO: Fit alpha with the Omega_p ansatz and fit also the residuals...

# TODO: first thing to do: assess the accuracy of using the parametric fit for alpha and beta. With only 5 numbers you may already have a pretty good model.

# TODO: symbolic regression using the phase as a parameter?

a,b = fit_L(L, beta[0], times)

amp_res, ph_res = get_amp_ph_beta_residuals(beta[0]-beta_guess(L, a, b), times)

M = (theta[0]+theta[1])
print('#############')
input_beta_for_alpha = ph_res
params_alpha = fit_alpha(alpha[0], input_beta_for_alpha, L, omega_orb, M, times)
res_alpha = alpha[0] -  get_alpha(input_beta_for_alpha, L, omega_orb, M, *params_alpha)

#alpha_res_trend, alpha_res_oscillation, ids_trend_model = fit_res_alpha_trend_EMD(res_alpha, times)
#ids_trend_model = slice(0, len(times))
#alpha_res_trend, alpha_res_oscillation = np.zeros(times.shape), np.zeros(times.shape)
#amp_alpha_res = np.mean(np.abs(scipy.signal.hilbert(alpha_res_oscillation)[ids_trend_model]))

params_res = fit_res_alpha_trend(res_alpha, times, input_beta_for_alpha, L, omega_orb, M)
alpha_res_trend = get_alpha_res(input_beta_for_alpha, L, omega_orb, M, *params_res)


if False:
	plt.plot(times, L/L[0])
	plt.plot(times, omega_orb/omega_orb[0], label = 'omega')
	plt.legend()
	plt.show()
	quit()

if False:
	res_alpha = beta[0] -beta_guess(L, a, b)
	res_alpha = alpha[0] -  get_alpha(input_beta_for_alpha, L, omega_orb, M, *params_alpha)

	ids_, = np.where(times<-0.1)
	t = times[ids_]
	emd = EMD()
	imf = emd(res_alpha[ids_], t)

	res_trend = imf[2]
	res_oscilallation = imf[1]

	amp_alpha_res, ph_alpha_res = get_amp_ph_beta_residuals(res_oscilallation, t)


	plt.figure()
	plt.plot(t, res_trend)
	plt.figure()
	plt.plot(t, res_oscilallation, label = 'oscillatory res')
	plt.plot(t, amp_alpha_res*np.cos(ph_alpha_res), label = 'alpha ph-amp')
	plt.plot(t, amp_alpha_res*np.sin(ph_res)[ids_], label = 'beta ph- alpha amp')
	plt.legend()
	plt.show()



	N = imf.shape[0]+1

	# Plot results
	plt.subplot(N,1,1)
	plt.plot(t, res_alpha[ids_], 'r')
	plt.xlabel("Time [s]")

	for n, imf in enumerate(imf):
		plt.subplot(N,1,n+2)
		plt.plot(t, imf, 'g')
		plt.title("imf "+str(n+1))
		plt.xlabel("Time [s]")
	plt.tight_layout()



	#plt.plot(times, res_alpha)
	#plt.ylim([-0.1,.1])
	#plt.gca().twinx().plot(times, np.cos(beta[0]), c= 'red')
	plt.show();quit()



fig, axes = plt.subplots(3,1, sharex = True)
axes[0].plot
axes[0].plot(times, beta_guess(L, a, b), c = 'cyan', label = 'fit')
axes[1].plot(times, beta[0]-beta_guess(L, a, b), c = 'coral')
axes[1].plot(times, amp_res*np.cos(ph_res), c = 'cyan')
axes[2].plot(times, beta[0], c = 'coral')
axes[2].plot(times, beta_guess(L, a, b)+amp_res*np.cos(ph_res), c = 'cyan')
axes[0].legend()

fig, axes = plt.subplots(3,1, sharex = True)
#pred_alpha = get_alpha(beta_guess(L, a, b), L, omega_orb, M, *params_alpha)
pred_alpha = get_alpha(input_beta_for_alpha, L, omega_orb, M, *params_alpha)
axes[0].plot(times, alpha[0], c = 'coral', label = 'true')
axes[0].plot(times, pred_alpha, c = 'cyan', label = 'fit')
axes[1].plot(times, alpha[0]-pred_alpha, c = 'coral')
axes[1].plot(times, alpha_res_trend, c = 'cyan')

#axes[2].plot(times, alpha_res_oscillation, c = 'red', label = 'alpha oscillation')
#axes[2].plot(times, amp_alpha_res*np.sin(ph_res), c = 'blue', label = 'beta ph- alpha amp')
#axes[2].legend()

axes[1].set_ylim([-1,1])
axes[0].legend()

plt.show()
quit()

plt.figure()
plt.title('tan beta')
plt.plot(times, tanbeta_, c='orange')
ax_t = plt.gca().twinx()
ax_t.plot(times, np.tan(beta[0]), c = 'blue')

plt.figure()
plt.title('beta')
plt.plot(times, beta[0], c = 'blue')
ax_t = plt.gca().twinx()
ax_t.plot(times, np.arccos(cosbeta_), c = 'green')

plt.show()
