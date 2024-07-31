import lalsimulation as lalsim
import lalsimulation.tilts_at_infinity as tilt_infty
import lal
import matplotlib.pyplot as plt
import numpy as np
import mlgw
import timeit
import tensorflow as tf
import precession
import joblib
from mlgw.precession_helper import angle_manager, angle_params_keeper

from train_reduced_angles_NN import CosinesLayer

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def augment(theta):
	q = np.squeeze(theta[:,0])
	frequencies = np.logspace(0, 1., 10)
	return np.column_stack([*theta.T, q**2, *[np.cos(f*q) for f in frequencies]])

lal_cmd = """
hp, hc = lalsim.SimInspiralChooseTDWaveform( 
		m1*lalsim.lal.MSUN_SI, #m1
		m2*lalsim.lal.MSUN_SI, #m2
		s1x, s1y, s1z,
		s2x, s2y, s2z,
		1*1e6*lalsim.lal.PC_SI, #distance to source (in pc)
		inclination, #inclination
		np.pi/2. - phiref, #phi ref
		0., #longAscNodes
		0., #eccentricity
		0., #meanPerAno
		deltaT, # time incremental step
		fstart, # lowest value of freq
		fref, #some reference value of freq
		lal.CreateDict(), #some lal dictionary
		approx #approx method for the model
		)
	"""

mlgw_cmd_NP = "gen.get_WF(theta[[0,1,4,7]], times, modes = None)"
mlgw_cmd = "gen.get_twisted_modes(theta, times, modes = None)"
angles_cmd = "lalsim.SimIMRPhenomTPHM_CoprecModes(m1*lal.MSUN_SI, m2*lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z, 1, inclination, deltaT, fstart, fref, phiref, lalparams, 0)"
angles_cmd_ML = "gen.get_alpha_beta_gamma(theta, times)"

angles_cmd_precession = "precession.integrator_orbav(Lhinitial,S1hinitial,S2hinitial,v,q,chi1,chi2)"

angle_cmd_simple_ODE = "solve_ivp(right_hand_side, t_grid[[0,-100]],  (alpha[0,0], beta[0,0]), t_eval = t_grid[:-100])"

########################

precver = 300
FS = 4
deltaT = 1/(4*2048.)
fstart = 11
fref = 11
m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, inclination, phiref = 15, 5, 0.6*np.cos(0.), 0.6*np.sin(0.), 0.1, 4e-8, 0., 0.3, 0., 0.
approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomTPHM")
#approx = lalsim.SimInspiralGetApproximantFromString("SEOBNRv4PHM")

	#Options for precession package
Lhinitial = np.array([0,0,1])
S1hinitial = np.array([s1x, s1y, s1z])
S2hinitial = np.array([s2x, s2y, s2z])
chi1 = np.linalg.norm(S1hinitial)
chi2 = np.linalg.norm(S2hinitial)
S1hinitial /= chi1
S2hinitial /= chi2
q = m2/m1
r = np.linspace(35, 1, 10000)
v = precession.eval_v(r)
ODEsolution = precession.integrator_orbav(Lhinitial,S1hinitial,S2hinitial,v,q,chi1,chi2)
t_grid_precession_package = ODEsolution[0,:,-1]*4.93e-6 *20.
t_grid_precession_package -= t_grid_precession_package[-1]

if not True:
	#out_dict = tilt_infty.calc_tilts_prec_avg_regularized.prec_avg_tilt_comp_vec_inputs(m1*lalsim.lal.MSUN_SI, m2*lalsim.lal.MSUN_SI, np.array([s1x, s1y, s1z]), np.array([s2x, s2y, s2z]), fref)
	#print(out_dict)

	t1, t2, phi12 = tilt_infty.hybrid_spin_evolution.get_tilts(s1x, s1y, s1z, s2x, s2y, s2z, 0., 0., 1.)
	print('tilts start ', t1, t2, phi12)

	out_dict = tilt_infty.hybrid_spin_evolution.calc_tilts_at_infty_hybrid_evolve(m1*lalsim.lal.MSUN_SI, m2*lalsim.lal.MSUN_SI,
			np.linalg.norm([s1x, s1y, s1z]),
			np.linalg.norm([s2x, s2y, s2z]),
			t1, t2, phi12,
			fref,
			version ='v2')
	
	print('tilts end ', out_dict['tilt1_inf'], out_dict['tilt2_inf'])

gen = mlgw.GW_generator('/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/dev/precession/precessing_model')
t_grid = np.linspace(-18, 0.01, int(18/deltaT))
#theta = np.array([m1, m2, s1x, s1y, s1z, s2x, s2y, s2z])
theta = np.array([m1, m2, 0.6*np.cos(0.5), 0.6*np.sin(0.5), s1z, s2x, s2y, s2z])

gen.get_twisted_modes(theta, t_grid, modes = None)
quit()

alpha, beta, gamma = gen.get_alpha_beta_gamma_IMRPhenomTPHM(theta, t_grid, f_ref = fref, f_start = fstart)
if np.linalg.norm([s2x, s2y])<1e-6:
	alpha -= np.arctan2(theta[3], theta[2])-np.arctan2(s1y, s1x)

ModeArray = lalsim.SimInspiralCreateModeArray()
for mode in [(2,2), (2,1), (3,3), (4,4), (5,5)]:
	lalsim.SimInspiralModeArrayActivateMode(ModeArray, mode[0], mode[1])

lalparams = lal.CreateDict()
lalsim.SimInspiralWaveformParamsInsertModeArray(lalparams, ModeArray)
lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lalparams, 0)
#lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(lalparams, precver)
#lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(lalparams, FS)
lalsim.SimInspiralWaveformParamsInsertPhenomXHMAmpInterpolMB(lalparams,  1)


hlmQAT, alphaT, cosbetaT, gammaT, af = lalsim.SimIMRPhenomTPHM_CoprecModes(m1*lal.MSUN_SI, m2*lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z, 1, inclination, deltaT, fstart, fref, phiref, lalparams, 0)
times = np.linspace(-len(alphaT.data.data)*deltaT, 0, len(alphaT.data.data))
manager = angle_manager(gen, t_grid, fref, fstart, beta_residuals = not True)
L, omega_orb = manager.get_L(theta)


	# Mock ODE solver
omega_interp = interp1d(t_grid, omega_orb)
L_interp = interp1d(t_grid, L)
def right_hand_side(t, y):
	alpha_dot = -omega_interp(t)/np.sin(y.T[1]) * L_interp(t)/np.sqrt(2+L_interp(t)**2)
	beta_dot = omega_interp(t) * L_interp(t)*np.cos(y.T[1])/np.sqrt(2+L_interp(t)**2)
	
	return np.stack([alpha_dot, beta_dot], axis = -1)
	
#res = solve_ivp(right_hand_side, t_grid[[0,-100]],  (alpha[0,0], beta[0,0]), t_eval = t_grid[:-100])
#print(res)


# Timing
n = 2
time_angles_lal = timeit.timeit(angles_cmd, globals = globals(), number = n)/n
time_mlgw_NP = timeit.timeit(mlgw_cmd_NP, globals = globals(), number = n)/n
time_mlgw = timeit.timeit(mlgw_cmd, globals = globals(), number = n)/n
time_lal = timeit.timeit(lal_cmd, globals = globals(), number = n)/n
time_angles_ML = timeit.timeit(angles_cmd_ML, globals = globals(), number = n)/n
time_angles_precession = timeit.timeit(angles_cmd_precession, globals = globals(), number = n)/n
time_angles_simple_ODE = np.nan #timeit.timeit(angle_cmd_simple_ODE, globals = globals(), number = n)/n


#FIXME: is it a fair comparison the one I made for precession?? i.e. Am I generating the angles right?
print("Time angles lal {} s\nTime angles ML {} s\nTime angles PRECESSION {} s\nTime angles simple ODE {} s\nTime mlgw NP {} s\nTime mlgw {} s\nTime lal {} s".format(time_angles_lal, time_angles_ML, time_angles_precession, time_angles_simple_ODE, time_mlgw_NP, time_mlgw, time_lal))

quit()

fig, axes = plt.subplots(2,1)
plt.suptitle('alpha')
axes[0].plot(times, alphaT.data.data, label = 'IMR')
axes[0].plot(t_grid, alpha, label = 'mlgw')
axes[0].legend()
alpha_ = np.interp(times, t_grid, alpha)
axes[1].plot(times, alphaT.data.data - alpha_)
plt.figure()
plt.title('beta')
plt.plot(times, cosbetaT.data.data, label = 'IMR')
plt.plot(t_grid, beta, label = 'mlgw')
#plt.plot(t_grid_precession_package, ODEsolution[0,:,2], label = 'PRECESSION') #WTF???
plt.legend()
plt.show()


