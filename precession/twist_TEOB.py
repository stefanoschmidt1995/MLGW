import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../mlgw_v2')

import GW_generator as gen
from GW_helper import compute_optimal_mismatch, generate_waveform_TEOBResumS

import lal
import lalsimulation as lalsim

import scipy.optimize, scipy.integrate, scipy.interpolate

from twist_IMR import twist_modes

#TODO: two issues:  1) The twist modes function works but it is not consistent with what happens in the GW_generator module (Works now!! Bad thing with alpha)
#					2) The mismatch between TEOB e mlgw for the NP modes is bad: is it the ML model or the TEOB low resolution?



if __name__ == "__main__":

	gen.list_models()
	g = gen.GW_generator(0)

	file_TEOB_angles = '/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/precession/TEOBResumS_angles/bbh_test/anglesint.txt'
	file_TEOB_22_T = '/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/precession/TEOBResumS_angles/bbh_test/hTlm_l2_m2.txt'
	file_TEOB_21_T = '/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/precession/TEOBResumS_angles/bbh_test/hTlm_l2_m1.txt'
	file_TEOB_22 = '/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/precession/TEOBResumS_angles/bbh_test/hlm_interp_l2_m2.txt'
	file_TEOB_21 = '/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/precession/TEOBResumS_angles/bbh_test/hlm_interp_l2_m1.txt'

	t_TEOB, alpha, beta, gamma = np.loadtxt(file_TEOB_angles).T

	M_sun = 1.*4.93e-6 #solar mass in seconds
	t = np.linspace(t_TEOB[0]*M_sun, t_TEOB[-1]*M_sun, 100000)

		#getting TEOB angles
	alpha = np.interp(t, t_TEOB*M_sun, alpha)
	beta = np.interp(t, t_TEOB*M_sun, beta)
	gamma = np.interp(t, t_TEOB*M_sun, gamma)

		#getting TEOB modes
	t_22, h_22_amp, h_22_ph = np.loadtxt(file_TEOB_22).T
	h_TEOB_22 = np.interp(t, t_22*M_sun, h_22_amp)*np.exp(1j*np.interp(t, t_22*M_sun, h_22_ph))

	t_21, h_21_amp, h_21_ph = np.loadtxt(file_TEOB_21).T 
	h_TEOB_21 = np.interp(t, t_21*M_sun, h_21_amp)*np.exp(1j*np.interp(t, t_21*M_sun, h_21_ph))

	t_22, h_22_amp_T, h_22_ph_T = np.loadtxt(file_TEOB_22_T).T 
	#h_TEOB_22_T = np.interp(t, t_22*M_sun, h_22_amp_T)*np.exp(1j*np.interp(t, t_22*M_sun, h_22_ph_T))
	h_TEOB_22_T = h_22_amp_T*np.exp(1j* h_22_ph_T)
	h_TEOB_22_T = np.interp(t,t_22*M_sun, h_TEOB_22_T)

	t_21, h_21_amp_T, h_21_ph_T = np.loadtxt(file_TEOB_21_T).T 
	#h_TEOB_21_T = np.interp(t, t_21*M_sun, h_21_amp_T)*np.exp(1j*np.interp(t, t_21*M_sun, h_21_ph_T))
	h_TEOB_21_T = h_21_amp_T*np.exp(1j*h_21_ph_T)
	h_TEOB_21_T = np.interp(t,t_21*M_sun, h_TEOB_21_T)

		#twist modes usign TEOB modes
	h_T_mlgw = twist_modes(g, h_TEOB_22, h_TEOB_21, alpha, beta, -gamma)
	h_T_mlgw = h_T_mlgw*np.exp(1j*np.pi)

		#twist modes using mlgw
	t_mlgw = t - t[np.argmax(np.interp(t,t_22*M_sun, h_22_amp))] 
	theta = np.array([[.66666667, .33333333,-0.43,.4,-0.2, 0.5,0.,0.3]])#,[20,15,0.,.3,.1,.1,0.,-0.2]]
	h_p_mlgw, h_c_mlgw = g.get_twisted_modes(theta, t_mlgw, [(2,2),(2,1)], 400., -0.00, None, None)#,res['x'])
	h_T_mlgw_obj = (h_p_mlgw +1j* h_c_mlgw)/(2./9.)

		#Twist modes using mlgw generated modes
	h_p_mlgw_NP, h_c_mlgw_NP = g.get_modes(g.get_NP_theta(theta), t_mlgw, [(2,2),(2,1)], out_type = 'realimag')
	h_T_mlgw_function = (h_p_mlgw_NP +1j* h_c_mlgw_NP)/(2./9.)
	h_T_mlgw_function[:,:,1] *= np.exp(1j*0.)
	h_T_mlgw_function = twist_modes(g, h_T_mlgw_function[0,:,0], h_T_mlgw_function[0,:,1], alpha, beta, -gamma)

		#get 22/21 mode from mlgw
	h_21_mlgw = np.squeeze(h_p_mlgw_NP[...,1]+1j*h_c_mlgw_NP[...,1])/(2./9.)
	h_22_mlgw = np.squeeze(h_p_mlgw_NP[...,0]+1j*h_c_mlgw_NP[...,0])/(2./9.)

		#computing NP mismatches
	F_21, ph_21 = compute_optimal_mismatch(h_21_mlgw, h_TEOB_21)
	F_22, ph_22 = compute_optimal_mismatch(h_22_mlgw, h_TEOB_22)
	h_21_mlgw *= np.exp(-1j*ph_21)
	h_22_mlgw *= np.exp(-1j*ph_22)
	print("(2,2) mismatch ", F_22, ph_22)
	print("(2,1) mismatch ", F_21, ph_21)

		#computing TEOB python WF
	times, h_p_TEOB, h_c_TEOB, hlm, t_m = generate_waveform_TEOBResumS(*g.get_NP_theta(theta)[0,:], 1, 0, 0, 
				f_min = 400, verbose = True, t_step = 1e-5, modes = [(2,2),(2,1)],
				path_TEOBResumS = '/home/stefano/teobresums/Python'
				)
	h_TEOB_21_py = np.interp(t_mlgw, times, hlm['0'][0]*np.exp(1j*hlm['0'][1]) )
	h_TEOB_22_py = np.interp(t_mlgw, times, hlm['1'][0]*np.exp(1j*hlm['1'][1]) )
	print("(2,2) mismatch - py ", *compute_optimal_mismatch(h_22_mlgw, h_TEOB_22_py))
	print("(2,1) mismatch - py",  *compute_optimal_mismatch(h_21_mlgw, h_TEOB_21_py))
	print("(2,2) mismatch - TEOB vs py ", *compute_optimal_mismatch(h_TEOB_22, h_TEOB_22_py))
	print("(2,1) mismatch - TEOB vs py ",  *compute_optimal_mismatch(h_TEOB_21, h_TEOB_21_py))

		######### THIS IS TO HAVE BETTER TEOB WFs
		#uncomment this to compare the results with the mlgw output
	#h_TEOB_22_T = h_T_mlgw[0,:,0]
	#h_TEOB_21_T = h_T_mlgw[0,:,1]
	#h_T_mlgw = h_T_mlgw_obj	

		######## PLOTTING PART

	### plotting NP modes
	plt.figure()
	plt.title("NP 21 modes ")
	plt.plot(t,h_21_mlgw.real, label = 'mlgw')
	plt.plot(t, h_TEOB_21.real, label = 'TEOB')
	plt.plot(t, h_TEOB_21_py.real, label = 'TEOB-py')
	plt.legend()

	plt.figure()
	plt.title("NP 22 modes ")
	plt.plot(t,h_22_mlgw.real, label = 'mlgw')
	plt.plot(t, h_TEOB_22.real, label = 'TEOB')
	plt.plot(t, h_TEOB_22_py.real, label = 'TEOB-py')
	plt.legend()

	### plotting TEOB vs mlgw-twisted TEOB
	###### Or TEOB vs mlgw-generated WF
	plt.figure()
	plt.plot(t, beta)
	plt.plot(t, h_TEOB_22_T.real, label = 'TEOB')
	plt.plot(t, h_T_mlgw[0,:,0].real, label = 'mlgw')
	plt.legend()

	plt.figure()
	plt.plot(t, beta)
	plt.plot(t, h_TEOB_21_T.real, label = 'TEOB')
	plt.plot(t, h_T_mlgw[0,:,1].real, label = 'mlgw')
	plt.legend()

	fig, ax = plt.subplots(2,1)
	plt.suptitle("ph 21")
	ph_TEOB = np.unwrap(np.angle(h_TEOB_21_T))
	ph_mlgw = np.unwrap(np.angle(h_T_mlgw[0,:,1]))
	ax[0].plot(t, ph_TEOB, label = 'TEOB')
	ax[0].plot(t, ph_mlgw, label = 'mlgw')
	ax[0].legend()
	ax[1].plot(t, ph_mlgw - ph_TEOB)

	fig, ax = plt.subplots(2,1)
	plt.suptitle("ph 22")
	ph_TEOB = np.unwrap(np.angle(h_TEOB_22_T))
	ph_mlgw = np.unwrap(np.angle(h_T_mlgw[0,:,0]))
	ax[0].plot(t, ph_TEOB, label = 'TEOB')
	ax[0].plot(t, ph_mlgw, label = 'mlgw')
	ax[0].legend()
	ax[1].plot(t, ph_mlgw - ph_TEOB)

	### checking for consistency between twist_modes and mlgw version
	fig, ax = plt.subplots(2,1)
	plt.suptitle("Consistency obj")
	ph_obj = np.unwrap(np.angle(h_T_mlgw_obj[0,:,:]), axis = 0)
	ph_mlgw = np.unwrap(np.angle(h_T_mlgw_function[0,:,:]), axis = 0)
	ax[0].plot(t, ph_obj[:,0]-ph_mlgw[:,0])
	ax[0].set_title('22')
	ax[1].plot(t, ph_obj[:,1]-ph_mlgw[:,1])
	ax[1].set_title('21')

	plt.show()
















