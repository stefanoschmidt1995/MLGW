"""
Module precession_helper.py
===========================
	Module for training a ML model for fitting the precessing angles alpha, beta as a function of (theta1, theta2, deltaphi, chi1, chi2, q).
	Requires precession module (pip install precession)
"""

import numpy as np
import precession
import os

def get_alpha_beta(q, chi1, chi2, theta1, theta2, delta_phi, r_0, times, verbose = False):
	"""
get_alpha_beta
==============
	Returns angles alpha and beta by solving PN equations for spins. Uses module precession.
	Angles are evaluated on a user-given time grid (units: s/M_sun) s.t. the 0 of time is at separation r = M_tot.
	Inputs:
		q (N,)				mass ratio (>1)
		chi1 (N,)			dimensionless spin magnitude of BH 1 (in [0,1])
		chi1 (N,)			dimensionless spin magnitude of BH 2 (in [0,1])
		theta1 (N,)			angle between spin 1 and L
		theta2 (N,)			angle between spin 2 and L
		delta_phi (N,)		angle between in plane projection of the spins
		r_0					initial separation (in natural units)
		times (D,)			times at which alpha, beta are evaluated (units s/M_sun)
		verbose 			whether to suppress the output of precession package
	Outputs:
		alpha (N,D)		alpha angle evaluated at times
		beta (N,D)		beta angle evaluated at times
	"""
	M_sun = 4.93e-6
	if isinstance(q,float):
		q = np.array(q)
		chi1 = np.array(chi1)
		chi2 = np.array(chi2)
		theta1 = np.array(theta1)
		theta2 = np.array(theta2)
		delta_phi = np.array(delta_phi)

	if len(set([q.shape, chi1.shape, chi2.shape, theta1.shape, theta2.shape, delta_phi.shape])) != 1:
		raise RuntimeError("Inputs are not of the same shape (N,). Unable to continue")

	if q.ndim == 0:
		q = q[None]
		chi1 = chi1[None]; chi2 = chi2[None]
		theta1 = theta1[None]; theta2 = theta2[None]; delta_phi = delta_phi[None]
		squeeze = True
	else:
		squeeze = False

		#initializing vectors
	alpha = np.zeros((q.shape[0],times.shape[0]))
	beta = np.zeros((q.shape[0],times.shape[0]))
	
	if not verbose:
		import sys, os	
		devnull = open(os.devnull, "w")
		old_stdout = sys.stdout
		sys.stdout = devnull
	else:
		old_stdout = sys.stdout

		#computing alpha, beta
	for i in range(q.shape[0]):
			#computing initial conditions for the time evolution
		q_ = 1./q[i] #using conventions of precession package
		M,m1,m2,S1,S2=precession.get_fixed(q_,chi1[i],chi2[i]) #M_tot is always set to 1

		#print(q_, chi1[i], chi2[i], theta1[i],theta2[i], delta_phi[i], S1, S2, M)
		old_stdout.write("Generated angle "+str(i)+"\n")
		old_stdout.flush()

		xi,J, S = precession.from_the_angles(theta1[i],theta2[i], delta_phi[i], q_, S1,S2, r_0) 

		J_vec,L_vec,S1_vec,S2_vec,S_vec = precession.Jframe_projection(xi, S, J, q_, S1, S2, r_0) #initial conditions given angles

		r_f = 1.*M
		sep = np.linspace(r_0, r_f, 10000)

		Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z, t = precession.orbit_vectors(*L_vec, *S1_vec, *S2_vec, sep, q_, time = True) #time evolution of L, S1, S2
		L = np.sqrt(Lx**2 + Ly**2 + Lz**2)
		
		temp_alpha = np.unwrap(np.arctan2(Ly,Lx))
		temp_beta = np.arccos(Lz/L)
		
		alpha[i,:] = np.interp(times, (t-t[-1])*M_sun, temp_alpha)
		beta[i,:] = np.interp(times, (t-t[-1])*M_sun, temp_beta)
	
	if not verbose:
		devnull.close()
		sys.stdout = old_stdout

	if squeeze:
		return np.squeeze(alpha), np.squeeze(beta)

	return alpha, beta


def create_dataset_alpha_beta(N_angles, filename, N_grid, tau_min, q_range, chi1_range= (0.,1.), chi2_range = (0.,1.), theta1_range = (0., np.pi), theta2_range = (0., np.pi), delta_phi_range = (-np.pi, np.pi) ):
	"""
create_dataset_alpha_beta
=========================
	Creates a dataset for the angles alpha and beta.
	The dataset consist in parameter vector (q, chi1, chi2, theta1, theta2, delta_phi) associated to two vectors alpha and beta.
	User must specify a time grid at which the angles are evaluated at.
	More specifically, data are stored in 3 vectors:
		param_vector	vector holding source parameters (q, chi1, chi2, theta1, theta2, delta_phi)
		alpha_vector	vector holding alpha angle for each source evaluated at some N_grid equally spaced points
		beta_vector		vector holding beta angle for each source evaluated at some N_grid equally spaced points
	The values of parameters are randomly drawn within the user given constraints.
	Dataset is saved to file, given in filename and can be loaded with load_angle_dataset.
	Inputs:
		N_angles			Number of angles to include in the dataset
		filename			Name of the file to save the dataset at
		N_grid				Number of grid points
		tau_min				Starting time at which the angles are computed (in s/M_sun)
		q_range				Tuple of values for the range in which to draw the q values. If a single value, q is fixed
		chi1_range			Tuple of values for the range in which to draw the dimensionless spin values of BH 1. If a single value, chi1 is fixed
		chi2_range			Tuple of values for the range in which to draw the dimensionless spin values of BH 2. If a single value, chi2 is fixed
		theta1_range		Tuple of values for the range in which to draw the angles between spin 1 and L. If a single value, theta1 is fixed
		theta2_range		Tuple of values for the range in which to draw the angles between spin 2 and L. If a single value, theta2 is fixed
		delta_phi_range		Tuple of values for the range in which to draw the angles between the in-plane components of the spins. If a single value, delta_phi_range is fixed
	"""
	if not isinstance(N_grid, int):
		raise TypeError("N_grid is "+str(type(N_grid))+"! Expected to be a int.")
	if not isinstance(filename, str):
		raise TypeError("filename is "+str(type(filename))+"! Expected to be a string.")

	range_list = [q_range, chi1_range, chi2_range, theta1_range, theta2_range, delta_phi_range]

	time_grid = np.linspace(-np.abs(tau_min), 0., N_grid)
		#initializing file. If file is full, it is assumed to have the proper time grid
	if not os.path.isfile(filename): #file doesn't exist: must be created with proper header
		filebuff = open(filename,'w')
		print("New file ", filename, " created")
		time_header = np.concatenate((np.zeros((6,)), time_grid, time_grid) )[None,:]
		np.savetxt(filebuff, time_header, header = "#Alpha, Beta dataset" +"\n# row: params (None,6) | alpha (None,"+str(N_grid)+")| beta (None,"+str(N_grid)+")\n# N_grid = "+str(N_grid)+" | tau_min ="+str(tau_min)+" | q_range = "+str(q_range)+" | chi1_range = "+str(chi1_range)+" | chi2_range = "+str(chi2_range)+" | theta1_range = "+str(theta1_range)+" | theta2_range = "+str(theta2_range)+" | delta_phi_range = "+str(delta_phi_range), newline = '\n')
	else:
		filebuff = open(filename,'a')
	#computing an approximate r_0 as a function of tau_min
	M_sun = 4.93e-6
	r_0 = 2.5 * np.power(tau_min/M_sun, .25) #look eq. 4.26 Maggiore
	
	#deal with the case in which ranges are not tuples
	for i, r in enumerate(range_list):
		if not isinstance(r,tuple):
			if isinstance(r, float):
				range_list[i] = (r,r)
			else:
				raise RuntimeError("Wrong type of limit given: expected tuple or float!")

	#creating limits for random draws
	lower_limits = [r[0] for r in range_list]	
	upper_limits = [r[1] for r in range_list]	
	
	b_size = 2 #batch size at which angles are stored before being saved
	count = 0 #keep track of how many angles were generated
	while True:
		if N_angles- count > b_size:
			N = b_size
		elif N_angles - count > 0:
			N = N_angles -count
		else:
			break

		params = np.random.uniform(lower_limits, upper_limits, (N, len(range_list))) #(N,6) #parameters to generate the angles at
		count += N

		alpha, beta = get_alpha_beta(*params.T, r_0, time_grid, False)
		to_save = np.concatenate([params, alpha, beta], axis = 1)
		np.savetxt(filebuff, to_save) #saving the batch to file
		print("Generated angle: ", count)

	return





















	


