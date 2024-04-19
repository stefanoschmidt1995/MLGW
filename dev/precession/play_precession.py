import lalsimulation as lalsim
import lalsimulation.tilts_at_infinity as tilt_infty
import lal
import matplotlib.pyplot as plt
import numpy as np
import mlgw
import timeit
import tensorflow as tf
import precession

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

mlgw_cmd = "gen.get_WF(theta[[0,1,4,7]], times, modes = None)"
angles_cmd = "lalsim.SimIMRPhenomTPHM_CoprecModes(m1*lal.MSUN_SI, m2*lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z, 1, inclination, deltaT, fstart, fref, phiref, lalparams, 0)"
angles_cmd_ML = "angle_model(np.array([[3]])); angle_model(np.array([[3]])); angle_model(np.array([[3]]))" #You are generating 3 angles!!

angles_cmd_precession = "precession.integrator_orbav(Lhinitial,S1hinitial,S2hinitial,v,q,chi1,chi2)"

########################

precver = 300
FS = 4
deltaT = 1/(4*2048.)
fstart = 10
fref = 15
m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, inclination, phiref = 50, 7, 0.6*np.cos(0.), 0.6*np.sin(0.), 0.1, 4e-8, 0., 0.3, 0., 0.
approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomTPHM")

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

gen = mlgw.GW_generator(0)
t_grid = np.linspace(-18, 0.01, int(18/deltaT))
#theta = np.array([m1, m2, s1x, s1y, s1z, s2x, s2y, s2z])
theta = np.array([m1, m2, 0.6*np.cos(0.5), 0.6*np.sin(0.5), s1z, s2x, s2y, s2z])
alpha, beta, gamma = gen.get_alpha_beta_gamma(theta, t_grid, f_ref = fref, f_start = fstart)
if np.linalg.norm([s2x, s2y])<1e-6:
	alpha -= np.arctan2(theta[3], theta[2])-np.arctan2(s1y, s1x)

angle_model = tf.keras.models.load_model('tmp_model')
in_ = np.array([[3], [2]])
angle = angle_model(in_)


gen.get_WF([[10,3, 0.4, -0.1],[10,3, 0.4, -0.1]], t_grid, modes = None)

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

# Timing
n = 2
time_angles_lal = timeit.timeit(angles_cmd, globals = globals(), number = n)/n
time_mlgw = timeit.timeit(mlgw_cmd, globals = globals(), number = n)/n
time_lal = timeit.timeit(lal_cmd, globals = globals(), number = n)/n
time_angles_ML = timeit.timeit(angles_cmd_ML, globals = globals(), number = n)/n
time_angles_precession = timeit.timeit(angles_cmd_precession, globals = globals(), number = n)/n

#FIXME: is it a fair comparison the one I made for precession?? i.e. Am I generating the angles right?
print("Time angles lal {} s\nTime angles ML {} s\nTime angles PRECESSION {} s\nTime mlgw {} s\nTime lal {} s".format(time_angles_lal, time_angles_ML, time_angles_precession, time_mlgw, time_lal))


fig, axes = plt.subplots(2,1)
plt.suptitle('alpha')
axes[0].plot(times, alphaT.data.data, label = 'IMR')
axes[0].plot(t_grid, alpha[0], label = 'mlgw')
axes[0].legend()
alpha_ = np.interp(times, t_grid, alpha[0])
axes[1].plot(times, alphaT.data.data - alpha_)
plt.figure()
plt.title('beta')
plt.plot(times, cosbetaT.data.data, label = 'IMR')
plt.plot(t_grid, beta[0], label = 'mlgw')
#plt.plot(t_grid_precession_package, ODEsolution[0,:,2], label = 'PRECESSION') #WTF???
plt.legend()
plt.show()


