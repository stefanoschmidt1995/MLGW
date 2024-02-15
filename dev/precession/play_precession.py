import lalsimulation as lalsim
import lalsimulation.tilts_at_infinity as tilt_infty
import lal
import matplotlib.pyplot as plt
import numpy as np
import mlgw
import timeit

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

########################

precver = 300
FS = 4
deltaT = 1/(4*2048.)
fstart = 10
fref = 15
m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, inclination, phiref = 50, 7, 0.6*np.cos(0.), 0.6*np.sin(0.), 0.1, 4e-8, 0., 0.3, 0., 0.
approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomTPHM")

if True:
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


hlmQAT, alphaT, betaT, gammaT, af = lalsim.SimIMRPhenomTPHM_CoprecModes(m1*lal.MSUN_SI, m2*lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z, 1, inclination, deltaT, fstart, fref, phiref, lalparams, 0)
times = np.linspace(-len(alphaT.data.data)*deltaT, 0, len(alphaT.data.data))

# Timing
n = 2
time_angles = timeit.timeit(angles_cmd, globals = globals(), number = n)/n
time_mlgw = timeit.timeit(mlgw_cmd, globals = globals(), number = n)/n
time_lal = timeit.timeit(lal_cmd, globals = globals(), number = n)/n

print("Time angles {} s\nTime mlgw {} s\nTime lal {} s".format(time_angles, time_mlgw, time_lal))


fig, axes = plt.subplots(2,1)
plt.suptitle('alpha')
axes[0].plot(times, alphaT.data.data, label = 'IMR')
axes[0].plot(t_grid, alpha[0], label = 'mlgw')
axes[0].legend()
alpha_ = np.interp(times, t_grid, alpha[0])
axes[1].plot(times, alphaT.data.data - alpha_)
plt.figure()
plt.title('beta')
plt.plot(times, betaT.data.data, label = 'IMR')
plt.plot(t_grid, beta[0], label = 'mlgw')
plt.legend()
plt.show()


