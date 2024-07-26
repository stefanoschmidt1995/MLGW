import glob
import matplotlib.pyplot as plt
import mlgw
from mlgw.precession_helper import angle_params_keeper, angle_manager, get_IMRPhenomTPHM_angles, Rot, get_alpha0_beta0_gamma0, to_cartesian
import numpy as np


def get_beta_from_deltaPhi(deltaPhi, phi1):
	s1, s2, t1, t2, phi1, phi2 = 0.72643506,.822, 1.40939778,1.54355588, phi1, phi1-deltaPhi
	s1x, s1y, s1z = to_cartesian(s1, t1, phi1)
	s2x, s2y, s2z =  to_cartesian(s2, t2, phi2)
	theta = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z]
	_, beta, _ = get_IMRPhenomTPHM_angles(*theta, 
		t_grid = t_grid, fstart = fstart, fref = fstart, inclination = iota, phiref = phi_ref)
	beta = np.arccos(beta)
	
	return theta, beta
#################################

gen = mlgw.GW_generator()

mtot = 20.
q = 7.92726497
m1, m2 = q * mtot / (1+q), mtot / (1+q)
iota, phi_ref = 0,0 #The start angles do not seem to depend on this!!
t_grid = np.linspace(-2*mtot, 0.001, int(2*mtot+0.001)*4096)


if False:
	deltaPhi = np.random.uniform(0,2*np.pi)
	phi1_bis = np.random.uniform(0,2*np.pi)
	
	s1, s2, t1, t2, phi1, phi2 = 0.72643506,.822, 1.40939778,1.54355588, 0., 0.
	s1_bis, s2_bis, t1_bis, t2_bis, phi1_bis, phi2_bis = 0.72643506,.822, 1.40939778,1.54355588, phi1_bis, phi1_bis - deltaPhi
	
	s1x, s1y, s1z = to_cartesian(s1, t1, phi1)
	s2x, s2y, s2z =  to_cartesian(s2, t2, phi2)
	
else:
	s1x, s1y, s1z = 0.7, 0, -0.3
	s2x, s2y, s2z = -0.5, 0, -0.2
	
	s1_bis, s2_bis, t1_bis, t2_bis, phi1_bis, phi2_bis = 0.72643506,.822, 1.40939778,1.54355588, 0,0

fstart, _ = gen.get_fref_angles([m1, m2, s1z, s2z])
manager = angle_manager(gen, t_grid, fstart, fstart, beta_residuals = not True)

if False:
	plt.figure()
	deltaPhi = 3*np.pi/2
	plt.title('deltaPhi = {}'.format(deltaPhi))
	for phi1 in np.linspace(0, 2*np.pi, 10):
		theta, beta = get_beta_from_deltaPhi(deltaPhi, phi1)
		Psi, _, s = manager.get_reduced_alpha_beta(theta)
		_, beta_trend, _ = manager.get_alpha_beta_gamma(theta, Psi)
		
		
		color = next(plt.gca()._get_lines.prop_cycler)['color']
		plt.plot(t_grid, beta, c = color, ls = '-')
		plt.plot(t_grid, beta_trend, c = color, ls = '--')
		
	plt.show()
	quit()

	
			#f_ref and f_start are expressed in terms of the 22 frequency
theta = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z]
alpha, beta, gamma = get_IMRPhenomTPHM_angles(*theta, t_grid = t_grid, fstart = fstart, fref = fstart, inclination = iota, phiref = phi_ref)
beta = np.arccos(beta)

theta_bis = [m1, m2, *to_cartesian(s1_bis, t1_bis, phi1_bis), *to_cartesian(s2_bis, t2_bis, phi2_bis)]
alpha_bis, beta_bis, gamma_bis = get_IMRPhenomTPHM_angles(*theta_bis, t_grid = t_grid, fstart = fstart, fref = fstart, inclination = iota, phiref = phi_ref)
beta_bis = np.arccos(beta_bis)


Psi, _, s = manager.get_reduced_alpha_beta(theta)
Psi = angle_params_keeper(Psi)
Psi_bis, _, s = manager.get_reduced_alpha_beta(theta_bis)
Psi_bis = angle_params_keeper(Psi_bis)


L, omega_orb = gen.get_L(theta, t_grid)
alpha0, beta0, gamma0 = alpha[0], beta[0], gamma[0]
alpha0_pred, beta0_pred, gamma0_pred = get_alpha0_beta0_gamma0(theta, L[0])
print(alpha0, beta0, gamma0)
print(alpha0_pred, beta0_pred, gamma0_pred)


print('Psi', Psi)
print('Psi_bis', Psi_bis)


_, beta0_pred, _ = get_alpha0_beta0_gamma0(theta, L[0])

fig, axes = plt.subplots(3,1)
axes[0].plot(t_grid, alpha_bis-alpha_bis[0])
axes[0].plot(t_grid, alpha-alpha[0])
axes[1].set_title('alpha - alpha_bis')
axes[1].plot(t_grid, alpha_bis-alpha_bis[0]-(alpha-alpha[0]), c= 'coral')
axes[2].set_title('cos beta')
axes[2].plot(t_grid, np.cos(beta_bis))
axes[2].plot(t_grid, np.cos(beta))
plt.tight_layout()


#ax_bis = axes[1].twinx()
#ax_bis.plot(t_grid, np.cos(beta))
#ax_bis.plot(t_grid, np.cos(beta_bis))

alpha_pred, beta_pred, _ = manager.get_alpha_beta_gamma(theta, Psi)
alpha_pred_bis, beta_pred_bis, _ = manager.get_alpha_beta_gamma(theta_bis, Psi_bis)

fig, axes = plt.subplots(3,1)
axes[0].set_title('alpha - alpha_pred')
axes[0].plot(t_grid, alpha-alpha_pred-alpha[0]+alpha_pred[0], label = 'alpha')
axes[0].plot(t_grid, alpha_bis-alpha_pred_bis-alpha_bis[0]+alpha_pred_bis[0], label = 'alpha_bis')
axes[0].legend()
axes[1].set_title('alpha_pred - alpha_pred_bis')
axes[1].plot(t_grid, alpha_pred-alpha[0]-(alpha_pred_bis-alpha_bis[0]))

axes[2].set_title('alpha_bis - alpha_pred')
axes[2].plot(t_grid, alpha_bis-alpha_bis[0]-(alpha_pred-alpha_pred[0]))
axes[2].set_ylim([-1,1])
plt.tight_layout()

plt.figure()
plt.plot(t_grid, beta_bis, c= 'cyan')
plt.plot(t_grid, beta_pred_bis, ls = '--', c= 'cyan' )
plt.plot(t_grid, beta_pred, ls = '--', c= 'coral')
plt.plot(t_grid, beta, c= 'coral')
plt.axhline(beta0_pred, ls = '--', c = 'k')

plt.figure()
plt.title('residuals together')
plt.plot(t_grid, beta-beta_pred, c= 'cyan', label = 'beta')
ax_twin = plt.gca().twinx()
ax_twin.plot(t_grid, (alpha-alpha_pred), c= 'coral', label = 'alpha')
s = np.max(np.abs((alpha-alpha_pred)[:4096*30]))*2
ax_twin.set_ylim([-s,s])
plt.legend()

plt.show()





####################
# Following: https://arxiv.org/abs/2004.06503

L_dim = L[0]*mtot**2
L_vect = np.array([0, 0, L_dim])
S1 = np.array([s1x, s1y, s1z])*m1**2
S2 = np.array([s2x, s2y, s2z])*m2**2
S = S1 + S2

J = L_vect + S
J_norm = J/np.linalg.norm(J)

theta_JL0 = np.arccos(J_norm[2]) #polar angle of J in the L0 frame
phi_JL0 = np.arctan2(J_norm[1], J_norm[0]) #azimuthal angle of J in the L0 frame

	#the L0 frame components of the WF propagation direction L
#Bullshit!!
N_hat = np.array([np.sin(iota)*np.cos(np.pi/2-phi_ref), np.sin(iota)*np.sin(np.pi/2-phi_ref), np.cos(iota)]) 
N_hat_Jprime = np.linalg.multi_dot([Rot(-theta_JL0, 'y'),Rot(-phi_JL0,'z'),N_hat])
kappa = np.arctan2(N_hat_Jprime[1], N_hat_Jprime[0]) #some angle I don't quite get...

theta_JN = np.arccos(np.dot(J_norm, N_hat))

#alpha offset
#https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomTPHM_EulerAngles.c#L1287
alpha_offset = np.arctan2(S[1], S[0]) - np.pi
kappa = np.pi-alpha_offset

L_Jframe =  np.linalg.multi_dot([Rot(-kappa, 'z'), Rot(-theta_JL0, 'y'),Rot(-phi_JL0, 'z'), np.array([0,0,1])])
#print('L_Jframe',L_Jframe)
#print(np.arccos(L_Jframe[0]))


#alpha0, beta0, gamma0 parametrize the rotation from the L0 frame to the J frame and viceversa

alpha0_pred = alpha_offset
beta0_pred = theta_JL0
gamma0_pred = np.pi-phi_JL0

print(alpha0, beta0, gamma0)
print(alpha0_pred, beta0_pred, gamma0_pred)

quit()
plt.figure()
plt.plot(t_grid, beta)
plt.axhline(beta0_pred, ls = '--', c = 'k')
plt.show()





























