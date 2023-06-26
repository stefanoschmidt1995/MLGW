import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as lalsim
import lal

import sys
sys.path.insert(0,'../mlgw_v2')
import GW_generator as gen
from GW_helper import compute_optimal_mismatch

sys.path.insert(0,'./IMRPhenomTPHM')
from run_IMR import *

def twist_modes(g, h_22, h_21, alpha, beta, gamma):
	l = 2
	alpha, beta, gamma = alpha[None,:], beta[None,:], gamma[None,:]

	m_modes_list = [(2,2),(2,1)]#, (2,-1), (2,-2)] #len = M'

		#genereting the non-precessing l-modes available
	h_NP_l = np.column_stack([h_22, h_21, np.conj(h_21), np.conj(h_22)])[None,:] #(N,D,M'')
	mprime_modes_list = [(2,2),(2,1), (2,-1), (2,-2)]
	
	#h_NP_l = np.column_stack([h_22, h_21, np.conj(h_22), np.conj(h_21)])[None,:] #(N,D,M'')
	#mprime_modes_list = [(2,2), (2,1), (2,-2), (2,-1)]
			
	D_mmprime = g._GW_generator__get_Wigner_D_matrix(l, [lm[1] for lm in m_modes_list], [lm[1] for lm in mprime_modes_list], alpha, beta, gamma) #(N,D,M, M'')

		#putting everything together
	h_P_l = np.einsum('ijlk,ijk->ijl', D_mmprime, h_NP_l) #(N,D,M)
	
		#global roation for moving everything in L0 frame
	#D_mmprime_L0 = g._GW_generator__get_Wigner_D_matrix(l,[lm[1] for lm in m_modes_list], [lm[1] for lm in m_modes_list],  -gamma[:,0], -beta[:,0], -alpha[:,0]) #(N,M,M)
	#h_P_l = np.einsum('ilk,ijk -> ijl', D_mmprime_L0, h_P_l)
	
	return h_P_l

#FIXME: the loading part does not work when only 1 expert is employed!!! You should add a dimension manually
#FIXME: why TEOB has such better performance in mismatches for the HMs?? It has a shitty ring-down!
#TODO: think on how to create a dataset of angles and how to train a model for the Euler angles
#TODO: check if you can reproduce the WF using Euler Angles in L0 frame, applying global transformation at the end of everything
	  #The problem is to recover the right reference angles, not trivial operation !!
	  #The change of frame is done here: https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomTPHM.c#L380  
#TODO: Understand the impact of alpha0 and gamma0: they seem to matter for the mismatch!
#TODO: try to predict the coalescence time: you will gain in health (maggiore, eq. 5.270)
#TODO: have a look at how the worsening of mismatch after twist depends on the original mismatch and parameters

if __name__ == "__main__":
		#to create a first attempt of dataset
	#create_angles_dataset(N = 10, f_min = None, M_tot = 20., quaternions = True)


	g = gen.GW_generator(3)
	#theta = np.array([[40, 20, .4, -0.1, .2, .3, -0.5, -0.1]])
	theta = np.array([[35, 20, -0.9, 0.04 , .2, -0.82, 0.35, -0.1]])
	f_min = 20.
	t_step = 1e-4
	t_grid, alpha, beta, gamma, h_22_NP, h_21_NP, h_22, h_21, h_2m1, h_2m2 = get_IMRPhenomTPHM_modes(*theta[0,:], f_min, t_step)
	print("Theta: ",theta[0,:])
	print("alpha0, gamma0 ", alpha[0], gamma[0])

		#mlgw P modes
	t_mlgw = t_grid -  t_grid[np.argmax(np.abs(h_22_NP))]
	h_P_mlgw_real,h_P_mlgw_imag  = g.get_twisted_modes(theta, t_mlgw, [(2,2),(2,1)], f_min, None, None, L0_frame = True)
	h_P_mlgw = h_P_mlgw_real+1j*h_P_mlgw_imag

		#NP mlgw modes
	h_NP_mlgw_real,h_NP_mlgw_imag  = g.get_modes(theta[:,[0,1,4,7]], t_mlgw, [(2,2),(2,1)], out_type='realimag')
	h_NP_mlgw = h_NP_mlgw_real+1j*h_NP_mlgw_imag

		#modes twisted by mlgw
	h_P_mlgw_T = twist_modes(g, h_22_NP, h_21_NP , alpha, beta, gamma) #here mlgw performs only the twist
	#h_P_mlgw_T = twist_modes(g, h_NP_mlgw[0,:,0], h_NP_mlgw[0,:,1] , alpha, beta, gamma) #here mlgw performs only the twist

	t_grid = t_grid - t_grid[np.argmax(h_22_NP)]

		#P choose TD modes (in L0 frame)
	hlm = lalsim.SimInspiralChooseTDModes(0.,
					t_step,
					theta[0,0]*lalsim.lal.MSUN_SI,
					theta[0,1]*lalsim.lal.MSUN_SI,
					theta[0,2],
					theta[0,3],
					theta[0,4],
					theta[0,5],
					0.,
					theta[0,7],
					f_min,
					f_min,
					1e6*lalsim.lal.PC_SI,
					lal.CreateDict(),
					5,			#lmax
					lalsim.IMRPhenomTPHM
				)
	prefactor = 4.7864188273360336e-20
	m1, m2 = theta[0,0], theta[0,1]
	nu =  np.divide(m1/m2, np.square(1+m1/m2))
	amp_prefactor = prefactor*(m1+m2)/1.*nu			
	h_22_P_lal = lalsim.SphHarmTimeSeriesGetMode(hlm, 2, 2).data.data/amp_prefactor
	h_21_P_lal = lalsim.SphHarmTimeSeriesGetMode(hlm, 2, 1).data.data/amp_prefactor
	t_grid_lal = np.linspace(0, len(h_22_P_lal)*t_step, len(h_22_P_lal))
	t_grid_lal = t_grid_lal- t_grid_lal[np.argmax(h_22_P_lal)]
	h_22_P_lal = np.interp(t_grid, t_grid_lal, h_22_P_lal)
	h_21_P_lal = np.interp(t_grid, t_grid_lal, h_21_P_lal)

		#computing mismatch (up to merger)
	id_merger = np.argmin(np.abs(t_grid))

	print("##########\nMismatches up to {}".format(np.array(['merger', 'ringdown'])[[id_merger !=-1, id_merger==-1]][0]))
	print("Mismatch NP WFs: 22, 21", compute_optimal_mismatch(h_NP_mlgw[0,:id_merger,0], h_22_NP[:id_merger])[0][0], compute_optimal_mismatch(h_NP_mlgw[0,:id_merger,1], h_21_NP[:id_merger])[0][0])
	print("Mismatch P j0 WFs: 22, 21", compute_optimal_mismatch(h_P_mlgw[0,:id_merger,0], h_22[:id_merger])[0][0], compute_optimal_mismatch(h_P_mlgw[0,:id_merger,1], h_21[:id_merger])[0][0])
	print("Mismatch P L0 WFs: 22, 21", compute_optimal_mismatch(h_P_mlgw[0,:id_merger,0], h_22_P_lal[:id_merger])[0][0], compute_optimal_mismatch(h_P_mlgw[0,:id_merger,1], h_21_P_lal[:id_merger])[0][0])

		#plotting
	#ignoring annoying real/imag warnings in plots
	import warnings
	warnings.filterwarnings("ignore")
		
	plt.figure()
	plt.title("angles")
	plt.plot(t_grid, alpha, label = 'alpha')
	plt.plot(t_grid, beta, label = 'beta')
	plt.plot(t_grid, gamma, label = 'gamma')
	plt.legend()
		
	fig, ax = plt.subplots(2,1)
	fig.suptitle("Non-precessing WFs")
	ax[0].set_title("22 NP")
	ax[0].plot(t_grid, h_NP_mlgw[0,:,0], label = 'mlgw NP')
	ax[0].plot(t_grid, h_22_NP, label = 'IMR NP')
	#ax[0].plot(t_grid, h_22, label = 'IMR P')
	ax[1].set_title("21 NP")
	ax[1].plot(t_grid, h_NP_mlgw[0,:,1], label = 'mlgw NP')
	ax[1].plot(t_grid, h_21_NP*np.exp(-1j*np.pi*0.5), label = 'IMR NP')
	#ax[1].plot(t_grid, h_21, label = 'IMR P')
	ax[1].legend()
	fig.tight_layout()

	fig, ax = plt.subplots(2,1)
	fig.suptitle("Precessing WFs - J0")
	ax[0].set_title("22 P")
	ax[0].plot(t_grid, h_P_mlgw[0,:,0], label = 'mlgw full')
	ax[0].plot(t_grid, h_P_mlgw_T[0,:,0], label = 'mlgw twist')
	ax[0].plot(t_grid, h_22, label = 'IMR - J0')
	#ax[0].plot(t_grid, h_22_P_lal, label = 'IMR - L0')
	ax[1].set_title("21 P")
	ax[1].plot(t_grid, h_P_mlgw[0,:,1], label = 'mlgw full')
	ax[1].plot(t_grid, h_P_mlgw_T[0,:,1], label = 'mlgw twist')
	ax[1].plot(t_grid, h_21, label = 'IMR - J0')
	#ax[1].plot(t_grid, h_21_P_lal, label = 'IMR - L0')
	ax[1].legend()
	fig.tight_layout()

	fig, ax = plt.subplots(2,1)
	fig.suptitle("Precessing WFs - L0")
	ax[0].set_title("22 P")
	ax[0].plot(t_grid, h_P_mlgw[0,:,0], label = 'mlgw')
	ax[0].plot(t_grid, h_22_P_lal, label = 'IMR - L0')
	ax[1].set_title("21 P")
	ax[1].plot(t_grid, h_P_mlgw[0,:,1], label = 'mlgw')
	ax[1].plot(t_grid, h_21_P_lal, label = 'IMR - L0')
	ax[1].legend()
	fig.tight_layout()


	if False:
		plt.figure()
		plt.title("ph diff")
		plt.plot(t_grid, np.unwrap(np.angle(h_NP_mlgw[0,:,0])) - np.unwrap(np.angle(h_22_NP)), label = '22')
		plt.plot(t_grid, np.unwrap(np.angle(h_NP_mlgw[0,:,1])) - np.unwrap(np.angle(h_21_NP)), label = '21')
		plt.legend()


	plt.show()
