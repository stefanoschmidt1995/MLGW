import lal
import matplotlib.pyplot as plt
import numpy as np
import mlgw
import scipy.optimize
import scipy.signal

from mlgw.precession_helper import angle_manager, to_J0_frame, angle_params_keeper, get_IMRPhenomTPHM_angles, compute_S_effective
from mlgw.GW_helper import load_dataset

gen = mlgw.GW_generator() #creating an istance of the generator (using default model)
M = 20.

if True:
	q = np.random.uniform(1,10)
	s1, s2 = np.random.uniform(0,1,2)
	t1, t2 = np.arccos(np.random.uniform(-1,1, 2))
	phi1, phi2 = 0,0 #np.random.uniform(0, 2*np.pi, 2)
		
	s1x, s1y, s1z = s1*np.sin(t1)*np.cos(phi1), s1*np.sin(t1)*np.sin(phi1), s1*np.cos(t1)
	s2x, s2y, s2z = s2*np.sin(t2)*np.cos(phi2), s2*np.sin(t2)*np.sin(phi2), s2*np.cos(t2)
	m1, m2 = M*q/(1+q), M/(1+q)
	theta = np.array([m1, m2, s1x, s1y, s1z, s2x, s2y, s2z])

else:
	q = 8
	theta = np.array([M*q/(1+q), M/(1+q), 0.9, 0.3, -0.2, -0.1, 0.6, -0.3]) #q = 2

	s1, s2 = 0.8, 0.4
	t1, t2 = 0.01, 2.3
	phi1, phi2 = 1.4, 0.5
	
		#Old anomalies for beta (with the previous model for beta)
	q, s1, s2, t1, t2, phi1, phi2 = [1.52436715, 0.8, 0.4, 0.60446821, 2.3, 1.4, 0.5] #flat beta
	q, s1, s2, t1, t2, phi1, phi2 =	[9.59, 0.8, 0.4, 3.10193568, 2.3, 1.4, 0.5] # Weird beta: transitional precession??
	#q, s1, s2, t1, t2, phi1, phi2 = [4.727435326150019, 0.8, 0.4, 2.089697197113566, 2.3, 1.4, 0.5] #Failure of the residual model

		#Anomalies for alpha
	#q, s1, s2, t1, t2, phi1, phi2 = [1.58182687,0.22507783,0.54340743,0.98508761,1.57793967,0.89582684,4.61680613] #S2_p>S1_p
	#q, s1, s2, t1, t2, phi1, phi2 = [2.98188059,0.148861,0.83264453,0.41992953,2.12945445,0.60971312,0.52438885] #S2_p>S1_p
		#Normal things for alpha!
	q, s1, s2, t1, t2, phi1, phi2 = [6.92726497,0.62643506,0.10468152,1.40939778,2.54355588,3.65671621,3.07855742] #S2_p<S1_p
	#q, s1, s2, t1, t2, phi1, phi2 = [9.16765466,0.20438633,0.55932627,1.3034392,1.37161716,3.75072587,2.39235398] #S2_p<S1_p
		
	#q, s1, s2, t1, t2, phi1, phi2 = [9.16765466,0.20438633,0.55932627,1.3034392,1.37161716, 0, 0] #S2_p<S1_p
		
	s1x, s1y, s1z = s1*np.sin(t1)*np.cos(phi1), s1*np.sin(t1)*np.sin(phi1), s1*np.cos(t1)
	s2x, s2y, s2z = s2*np.sin(t2)*np.cos(phi2), s2*np.sin(t2)*np.sin(phi2), s2*np.cos(t2)
	m1, m2 = 20*q/(1+q), 20/(1+q)
	
	theta = np.array([m1, m2, s1x, s1y, s1z, s2x, s2y, s2z])

	#theta = np.array([15*q/(1+q), 15/(1+q), -0.6, 0, -0.2, 0.7, 0, 0.3])


times = np.linspace(-40, 0.02, 100000)

fstart = 2*gen.get_orbital_frequency([m1, m2, s1z, s2z], -40+1e-3, 1e-3)
fstart *= 0.99
print(fstart)

manager = angle_manager(gen, times, fstart, fstart, beta_residuals = not True)

Psi, alpha_res, success = manager.get_reduced_alpha_beta(theta, plot = True)
Psi = angle_params_keeper(Psi)

print('Success?', success)
print(Psi)

quit()

theta_scatter, targets_scatter, _ = load_dataset('datasets/angle_dataset_only_qs1s2t1t2.dat', N_entries = 1, N_grid = None, shuffle = False, n_params = 8)
#theta_scatter, targets_scatter, _ = load_dataset('datasets/angle_dataset_full.dat', N_entries = 1, N_grid = None, shuffle = False, n_params = 8)
#theta_scatter_onlyq, targets_scatter_onlyq, _ = load_dataset('datasets/angle_dataset_only_q_new_beta_model.dat', N_entries = 1, N_grid = None, shuffle = False, n_params = 8)
#theta_scatter, targets_scatter, _ = load_dataset('datasets/angle_dataset_only_qt1.dat', N_entries = 1, N_grid = None, shuffle = False, n_params = 7)
#theta_scatter_onlyq, targets_scatter_onlyq, _ = load_dataset('datasets/angle_dataset_only_q.dat', N_entries = 1, N_grid = None, shuffle = False, n_params = 7)

Psi_scatter =  angle_params_keeper(targets_scatter)
#Psi_scatter_onlyq =  angle_params_keeper(targets_scatter_onlyq)

PCA_theta, PCA_beta = np.loadtxt('old_pca_shit/pca_model_qs1s2t1t2/PCA_train_theta_angles.dat'), np.loadtxt('old_pca_shit/pca_model_qs1s2t1t2/PCA_train_beta.dat')
		
#ids_, = np.where(Psi_scatter.A_alpha>1)
#ids_, = np.where( (theta_scatter[:,3]>3.) | (theta_scatter[:,3]<0.5) |
#	(theta_scatter[:,1]<0.1) | (theta_scatter[:,1]<0.1) |
#	(theta_scatter[:,0]<1.3))

#print(theta_scatter[ids_][:3])
s1, s2, t1, t2, phi1, phi2 = theta_scatter[:,1:-1].T
s1x, s1y, s1z = s1*np.sin(t1)*np.cos(phi1), s1*np.sin(t1)*np.sin(phi1), s1*np.cos(t1)
s2x, s2y, s2z = s2*np.sin(t2)*np.cos(phi2), s2*np.sin(t2)*np.sin(phi2), s2*np.cos(t2)
S1_p, S2_p = compute_S_effective(20*theta_scatter[:,0]/(1+theta_scatter[:,0]), 20/(1+theta_scatter[:,0]),
	np.stack([s1x, s1y], axis = 1), np.stack([s2x, s2y], axis = 1))
chi_P_2D_norm = np.linalg.norm(S1_p + S2_p, axis =1)

color = chi_P_2D_norm
#color = theta_scatter[:,3]

ids_, = [], #np.where((chi_P_2D_norm<.1) | (theta_scatter[:,1]<0.1))

print(theta_scatter[ids_][:2])

plt.figure()
sc = plt.scatter(Psi_scatter.A_beta, Psi_scatter.B_beta, s =2, c = color)
#plt.scatter(Psi_scatter_onlyq.A_beta, Psi_scatter_onlyq.B_beta, s =2)
plt.scatter(Psi_scatter.A_beta[ids_], Psi_scatter.B_beta[ids_], s =2, c='r')
plt.scatter(Psi.A_beta, Psi.B_beta, s =40, marker = '*', c= 'r')
#plt.scatter(PCA_beta[:,0],  PCA_beta[:,1], s = 2,  label = 'true')
#plt.xlim([-0.5, 10])
#plt.ylim([-0.5, 10])
plt.colorbar(sc)
plt.xlabel('A_beta')
plt.ylabel('B_beta')

#ids_sort = np.argsort(theta_scatter_onlyq[:,0])

#fig, axes = plt.subplots(2,1, sharex = True)
#axes[0].plot(theta_scatter_onlyq[ids_sort, 0], Psi_scatter_onlyq.A_beta[ids_sort], lw = 1,  label = 'true')
#axes[1].plot(theta_scatter_onlyq[ids_sort, 0], Psi_scatter_onlyq.B_beta[ids_sort], lw = 1,  label = 'true')

plt.figure()
sc = plt.scatter(Psi_scatter.A_alpha, Psi_scatter.B_alpha, s =2, c = color)
plt.scatter(Psi_scatter.A_alpha[ids_], Psi_scatter.B_alpha[ids_], s =2, c= 'r')
plt.scatter(Psi.A_alpha, Psi.B_alpha, s =40, marker = '*', c= 'r')
plt.xlabel('A_alpha')
plt.ylabel('B_alpha')
plt.colorbar(sc)
plt.show()


quit()

	#Cheking that beta model makes sense: spoiler, it doesn't

alpha0, beta0, gamma0 = manager.get_angles_at_ref_frequency(theta)
L0_spins = theta[[2,3,4]] + theta[[5,6,7]]
J0_spins = to_J0_frame(L0_spins, alpha0, beta0, gamma0)
S_perp, S_par = np.linalg.norm(J0_spins[[0,1]]), np.abs(J0_spins[2])

print('{} -> {}'.format(L0_spins, J0_spins))

#theta_of_q = lambda q: np.array([15*q/(1+q), 15/(1+q), 0.9, 0.3, -0.2, -0.1, 0.6, -0.3])

#for q_ in np.linspace(1,12,13):
#	print(q_, manager.get_angles_at_ref_frequency(theta_of_q(q_)))

Psi = angle_params_keeper(Psi)

print(Psi.A_beta, Psi.B_beta)
print(S_perp, S_par)

alpha, cosbeta, gamma = get_IMRPhenomTPHM_angles(*theta, t_grid = manager.times, fref = manager.fref, fstart = manager.fref)

L, _ = manager.get_L(theta)
beta_fit = manager.get_beta_trend(L, Psi.A_beta, Psi.B_beta)
beta_spins = manager.get_beta_trend(L, S_perp, S_par)

plt.figure()
plt.plot(manager.times, np.arccos(cosbeta), label = 'true')
plt.plot(manager.times, beta_fit, label = 'fit')
plt.plot(manager.times, beta_spins, label = 'spins')
plt.legend()
plt.show()

#manager.get_beta_from_alpha(theta)

#print(Psi)
