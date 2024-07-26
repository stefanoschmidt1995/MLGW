import numpy as np
import matplotlib.pyplot as plt
import precession
import mlgw
from mlgw.precession_helper import angle_params_keeper, angle_manager, get_IMRPhenomTPHM_angles, Rot, get_alpha0_beta0_gamma0, to_cartesian
import scipy.optimize

gen = mlgw.GW_generator()

mtot = 20.
q = 5.092726497
s1, s2, t1, t2, phi1, phi2 = 0.372643506, .1822, 2.40939778,1.54355588, 1., 0

#q, s1, s2, t1, t2, phi1, phi2 =	[9.59, 0.8, 0.4, 3.10193568, 2.3, 1.4, 0.5]

deltaphi = -phi1 + phi2 #FIXME: decide how you define deltaPhi!!!
m1, m2 = q * mtot / (1+q), mtot / (1+q)
eta = m1*m2/mtot**2

s1x, s1y, s1z = to_cartesian(s1, t1, phi1)
s2x, s2y, s2z =  to_cartesian(s2, t2, phi2)
theta = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z]

assert np.allclose(precession.vectors_to_angles([0,0,1], [s1x, s1y, s1z], [s2x, s2y, s2z])[-1], deltaphi)

t_grid = np.linspace(-40, 0.001, int(40.001*4096.))
ids_, = np.where(t_grid<-0.05)
dt = t_grid[1]-t_grid[0]
fstart, _ = gen.get_fref_angles(theta)
manager = angle_manager(gen, t_grid, fstart, fstart, True)
L, omega_orb = gen.get_L(theta, t_grid)
	
alpha0_pred, beta0_pred, gamma0_pred = get_alpha0_beta0_gamma0(theta, L[0])


	#Computing beta with precession package
L_0 = L[0]
r = np.square(L_0/eta)
deltachi, kappa, chieff = precession.angles_to_conserved(theta1=t1, theta2=t2, deltaphi=deltaphi, r=r, q=1/q, chi1=s1, chi2=s2)

assert np.allclose(L_0, precession.eval_L(r, 1/q))
assert np.allclose(deltachi, (s1*np.cos(t1)*q -s2*np.cos(t2))/(1+q))
assert np.allclose(chieff, (m1*s1z+m2*s2z)/mtot)

deltachi_minus,deltachi_plus = precession.deltachilimits(kappa=kappa,r=r,chieff=chieff,q=1/q,chi1=s1,chi2=s2)

beta_precession = precession.eval_thetaL(deltachi, kappa, r, chieff, 1/q)
nutation_amp = precession.eval_delta_theta(kappa, r, chieff, 1/q, s1,s2)

assert np.allclose(beta0_pred, beta_precession)
print('nutation amp = ', nutation_amp)
print('beta prec', beta_precession)

	#Computing the WF

#alpha, beta, gamma, t_grid_IMR = get_IMRPhenomTPHM_angles(*theta, t_grid = None, fstart = fstart, fref = fstart)
alpha, beta, gamma = get_IMRPhenomTPHM_angles(*theta, t_grid = t_grid, fstart = fstart, fref = fstart)
beta = np.arccos(beta)

alpha0, beta0, gamma0 = alpha[0], beta[0], gamma[0]

#assert np.allclose(alpha0_pred, alpha0)
#assert np.allclose(beta0_pred, beta0, rtol = 0.1, atol = 0)
#assert np.allclose(gamma0_pred, gamma0)

	#Computing the approximation
Psi, _, _ = manager.get_reduced_alpha_beta(theta)
Psi = angle_params_keeper(Psi)
print(nutation_amp, Psi.amp_beta)
Psi.amp_beta = 0.
alpha_trend, beta_trend, _ = manager.get_alpha_beta_gamma(theta, Psi)

	#Computing beta(t) with precession pacakge - only one parameter!!!
def get_beta_trend_precession(A):
	#L_PN = L * (1+0.5* (omega_orb*mtot*4.93e-6)**(2/3)*(1-3*eta) + (3+eta)/np.square((L+B)/eta))
	r_of_t = np.square((L+A)/eta)
	kappa_of_t = precession.eval_kappa(theta1=t1, theta2=t2, deltaphi=deltaphi, r= r_of_t, q=1/q, chi1=s1, chi2=s2)
	return precession.eval_thetaL(deltachi, kappa_of_t, r_of_t, chieff, 1/q)

loss = lambda x: np.mean(np.square(get_beta_trend_precession(*x) - beta)[ids_])

res = scipy.optimize.minimize(loss, x0 = np.array([0]))

print(res)
beta_trend_precession = get_beta_trend_precession(*res.x)
A_beta = res.x

	#Computing alpha(t)
def get_Omega_p(L, q, A, B):
	r_of_t = np.square(L/eta)
	
	#return (3+1.5/q)*np.sqrt(L**2+A*L+total_S**2)/r_of_t**3
	return (3+1.5/q)*eta**6*np.sqrt(np.abs(L**2+A*L+total_S**2))/L**6
	
	#return  (3+1.5/q)*eta**6*np.sqrt(np.abs(L**2+A*L+B))/L**6
	
	return (3+1.5/q)*eta**6*(A*L+B)/L**6
	

def get_alpha_trend_precession(A,B):
	
	#r_of_t = np.square((A*L+B)/eta)
	#kappa_of_t = precession.eval_kappa(theta1=t1, theta2=t2, deltaphi=deltaphi, r= r_of_t, q=1/q, chi1=s1, chi2=s2)
	#Omega_p = precession.eval_OmegaL(deltachi, kappa_of_t, r_of_t, chieff, 1/q, s1, s2)/(mtot*4.93e-6)
	
	Omega_p = get_Omega_p(L, q, A, B)/(mtot*4.93e-6)
	
	return np.cumsum(Omega_p)*dt

total_S = precession.eval_S(deltachi = deltachi, kappa = kappa, r = r, chieff = chieff, q = 1/q)[0]
print(total_S)
A_term_alpha = 2*s1*(q/(1+q))**2*np.cos(t1)+ 2*s2/(1+q)**2*np.cos(t2)
print(A_term_alpha)

loss = lambda x: np.mean(np.square(get_alpha_trend_precession(*x)[ids_] - (alpha[ids_]-alpha[0])))

res = scipy.optimize.minimize(loss, x0 = np.array([A_term_alpha,total_S**2]))
#res = scipy.optimize.minimize(loss, x0 = np.array([0.01, 0]))
print('###########\n',res)

alpha_trend_precession = get_alpha_trend_precession(*res.x)
#alpha_trend_precession = get_alpha_trend_precession(A_term_alpha)
alpha_trend_precession = alpha_trend_precession -alpha_trend_precession[0]+alpha[0]

	#Plotting
plt.figure()
plt.plot(t_grid, alpha)
plt.plot(t_grid, alpha_trend, label = 'std')
plt.plot(t_grid, alpha_trend_precession, label = 'prec package')
plt.legend()

plt.figure()
plt.plot(t_grid, alpha - alpha_trend, label = 'std')
plt.plot(t_grid, alpha - alpha_trend_precession, label = 'prec package')
plt.legend()



plt.figure()
plt.plot(t_grid, beta)
plt.plot(t_grid, beta_trend, label = 'std')
plt.plot(t_grid, beta_trend_precession, label = 'prec package')
plt.axhline(beta_precession, ls = '--', c= 'k')
plt.plot(t_grid, beta_trend+nutation_amp, ls = 'dotted', c= 'k')
plt.plot(t_grid, beta_trend-nutation_amp, ls = 'dotted', c= 'k')
plt.legend()

plt.figure()
plt.plot(t_grid, beta - beta_trend, label = 'std')
plt.plot(t_grid, beta - beta_trend_precession, label = 'prec package')
plt.legend()

plt.show()








