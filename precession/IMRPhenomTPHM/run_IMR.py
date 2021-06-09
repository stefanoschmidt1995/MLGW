import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os


def plot():
	angles = np.loadtxt('angles.txt').T
	modes_P = np.loadtxt('modes_P.txt', dtype = np.complex128).T
	modes_NP = np.loadtxt('modes_NP.txt', dtype = np.complex128).T

	print("Time grid ", len(angles[:,0]))

	plt.figure()
	plt.title("Angles")
	plt.plot(angles[:,0],angles[:,1])
	plt.plot(angles[:,0],angles[:,2])
	plt.plot(angles[:,0],angles[:,3])

	plt.figure()
	plt.title("modes")
	plt.plot(modes_P[:,0], modes_P[:,31])
	plt.plot(modes_P[:,0], modes_NP[:,31])
	plt.plot(modes_P[:,0], modes_P[:,29])
	plt.plot(modes_P[:,0], modes_NP[:,29])
	plt.show()


def get_IMRPhenomTPHM_modes(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, f_min, delta_T):
	"Returns: t_grid, alpha, beta, gamma, h_22_NP, h_21_NP, h_22, h_21, h_2m1, h_2m2 "
	#subprocess.run("/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/precession/IMRPhenomTPHM/run_IMR {} {} {} {} {} {} {} {} {} {}".format(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, f_min, delta_T))
	os.system("/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/precession/IMRPhenomTPHM/run_IMR {} {} {} {} {} {} {} {} {} {}".format(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, f_min, delta_T))
	angles = np.loadtxt('angles.txt').T
	modes_P = np.loadtxt('modes_P.txt', dtype = np.complex128).T
	modes_NP = np.loadtxt('modes_NP.txt', dtype = np.complex128).T
	
	#os.system("rm -f {} {} {} {}".format('angles.txt', 'modes_P.txt', 'modes_NP.txt', 'header.txt' )) #should be useful eventually!

	return angles[:,0], angles[:,1], np.arccos(angles[:,2]), angles[:,3], modes_NP[:,29], modes_NP[:,31], modes_P[:,29], modes_P[:,31], modes_P[:,28], modes_P[:,30] #t_grid, alpha, beta, gamma, h_22_NP, h_21_NP, h_22, h_21, h_2m1, h_2m2 	

def get_IMRPhenomTPHM_angles(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, f_min, delta_T):
	"Returns: t_grid, alpha, beta, gamma"
	os.system("/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/precession/IMRPhenomTPHM/run_IMR {} {} {} {} {} {} {} {} {} {}".format(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, f_min, delta_T))
	angles = np.loadtxt('angles.txt').T
	h22_NP = (np.loadtxt('modes_NP.txt', dtype = np.complex128).T)[:,29]
		
	os.system("rm -f angles.txt modes_P.txt modes_NP.txt header.txt") #should be useful eventually!

	t = angles[:,0]
	t = t - t[np.argmax(np.abs(h22_NP))]

	return t , angles[:,1], np.arccos(angles[:,2]), angles[:,3] #t_grid, alpha, beta, gamma

def create_angles_dataset(N = 10, f_min = 20., quaternions = False):
	"First attempt of creating a dataset"
	from scipy.spatial.transform import Rotation as R


	fig_alpha, ax_alpha = plt.subplots(1,1)
	fig_beta, ax_beta = plt.subplots(1,1)
	fig_gamma, ax_gamma = plt.subplots(1,1)
	
	if quaternions: fig_quat, ax_quat = plt.subplots(4,1, figsize=(10,10))
	
	for i in range(N):
		m = np.random.uniform(10,100,2)
		theta = np.random.uniform(0,np.pi,2)
		phi = np.random.uniform(0,2*np.pi,2)
		S = np.random.uniform(0.1,.9,(2,))
		S = np.array([S*np.sin(theta)*np.cos(phi), S*np.sin(theta)*np.sin(phi), S*np.cos(theta)]).T
		
		t, alpha, beta, gamma = get_IMRPhenomTPHM_angles(*m, *S[0,:], *S[1,:], f_min, 1e-4 )
		
			#performing the scaling of the grid
		mchirp_5_3 = (m[0]*m[1])/(m[0]+m[1])**(1./3) #mchirp^(5/3)
		tau = 2.18* (1.21)**(5./3.)/mchirp_5_3 * (100/f_min)**(8./3.)
		print("Time to merger: N vs True ",tau, -t[0])
		tau = -t[0]
		
		ax_alpha.plot(t/tau, alpha)
		ax_beta.plot(t/tau, beta)
		ax_gamma.plot(t/tau, gamma)

			#doing quaternions
		if quaternions:
			rotation = R.from_euler('zyx', np.column_stack([alpha,beta,gamma]), degrees=False)
			quat = rotation.as_quat()
			for j in range(quat.shape[1]):
				ax_quat[j].plot(t/tau, quat[:,j], label = "quat {}".format(j))

		#ax_gamma.axvline(tau)
		#ax_beta.axvline(tau)
		#ax_alpha.axvline(tau)
	plt.show()
		
	return
















if __name__ == "__main__":
	plot()
