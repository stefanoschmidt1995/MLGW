"""
How to integrate stuff.
orbit_vector = integrates the 10 dimensional equations

Jframe_projection = given x_i, total J, total Spin, q computes a triad L, S1, S2 in cartesian coordinates

NN: small parametrization for initial conditions ----> Jframe_projection for setting initial conditions ---> time evolution of L, S1, S2

model: full L, S1, S2 ----> small parametrization for initial conditions ----> NN ----> time evolution of L, S1, S2 ----> a,beta, gamma

"""

import precession
import numpy as np
import time
import matplotlib.pyplot as plt
import integrate_ODE
import scipy

print("##########################")
fig0, ax0 = plt.subplots(2,1, sharex = True)
fig1, ax1 = plt.subplots(1,1)
fig2, ax2 = plt.subplots(2,1,sharex = True)
fig2.suptitle(r"$\alpha, \beta$")
fig0.suptitle(r"$S, \beta$")


t0=time.time()
for i in range(10):
	q=np.random.uniform(0.5,0.9) # Mass ratio
	chi1=np.random.uniform(0.,1.) # Primary’s spin magnitude
	chi2=np.random.uniform(0.,1.) # Secondary’s spin magnitude
	print("Take a BH binary with q={}, chi1={} and chi2={}".format(q,chi1,chi2))

	M,m1,m2,S1,S2=precession.get_fixed(q,chi1,chi2)
	print("Masses and spins: ",M,m1,m2,S1,S2)

		#separations
	ri=500*M  # Initial separation.
	rf=10.*M   # Final separation.
	sep=np.linspace(ri,rf,10000) # Output separations

		#random angles
	t1=np.random.uniform(0, np.pi) # theta 1
	t2= np.random.uniform(0, np.pi) # theta 2
	dp= np.random.uniform(-np.pi, np.pi) #delta phi
	xi,J, S = precession.from_the_angles(t1,t2, dp, q,S1,S2, sep[0])
	
	Jvec,Lvec,S1vec,S2vec,Svec = precession.Jframe_projection(xi, S, J, q, S1, S2, sep[0]) #initial conditions given angles

	print("Initial conditions: ",Jvec, Lvec, Svec)

	good_couple=(J>=min(precession.J_allowed(xi,q,S1,S2,sep[0])) and J<=max(precession.J_allowed(xi,q,S1,S2,sep[0])))
	if not good_couple:
		print("Wrong initial conditions")
		continue

	Lx_fvals, Ly_fvals, Lz_fvals, S1x_fvals, S1y_fvals, S1z_fvals, S2x_fvals, S2y_fvals, S2z_fvals, t_fvals = precession.orbit_vectors(*Lvec, *S1vec, *S2vec, sep, q, time = True)

		#precession avg
	#t1_prec,t2_prec,dp_prec = precession.evolve_angles(t1,t2, sep[0], sep, q,S1,S2)



		#computing J
	Jx_fvals = Lx_fvals+ S1x_fvals + S2x_fvals
	Jy_fvals = Ly_fvals+ S1y_fvals + S2y_fvals
	Jz_fvals = Lz_fvals+ S1z_fvals + S2z_fvals

	L_fvals = np.sqrt(Lx_fvals**2 + Ly_fvals**2 + Lz_fvals**2)
	J_fvals = np.sqrt(Jx_fvals**2 + Jy_fvals**2 + Jz_fvals**2)

	print(Jx_fvals, Jy_fvals, Jz_fvals)

	beta = np.arccos(Lz_fvals/L_fvals)

	ax0[0].plot(t_fvals-t_fvals[-1], np.linalg.norm(np.column_stack([S1x_fvals+S2x_fvals, S1y_fvals+S2y_fvals, S1z_fvals+S2z_fvals]),axis = 1))
	ax0[1].plot(t_fvals-t_fvals[-1], beta)

	#ax1.plot(t_fvals, np.arccos(Lz_fvals/L_fvals))
	#ax1.plot(t_fvals-t_fvals[-1], np.unwrap(np.angle(Lx_fvals+1j*Ly_fvals)))

	#beta = Lx_fvals*Jx_fvals+Ly_fvals*Jy_fvals+Lz_fvals*Jz_fvals
	#beta = np.divide(beta, np.multiply(L_fvals, J_fvals))
	#plt.plot(t_fvals, np.arccos(beta))

	#t1_orb = np.zeros((len(Lx_fvals),))
	#for i in range(len(Lx_fvals)):
	#	t1_orb[i], t2, deltaPhi, t12 = precession.build_angles(np.array(Lx_fvals[i],Ly_fvals[i],Lz_fvals[i]), np.array(S1x_fvals[i],S1y_fvals[i],S1z_fvals[i]),np.array(S2x_fvals[i],S2y_fvals[i],S2z_fvals[i]))
		
	ax2[0].plot(t_fvals-t_fvals[-1], np.unwrap(np.arctan2(Ly_fvals,Lx_fvals)))
	ax2[1].plot(t_fvals-t_fvals[-1], beta)

		#getting the extrema
	from scipy.signal import argrelextrema
	args = argrelextrema(np.arccos(Lz_fvals/L_fvals), np.greater)
	ax1.plot(t_fvals[args]-t_fvals[-1], np.arccos(Lz_fvals/L_fvals)[args])

	#beta_c = beta - 1j*beta
	#ax1.plot(t_fvals-t_fvals[-1], np.unwrap(np.angle(beta_c)))
	

	t=time.time()-t0
	print("Executed in {}s".format(t))

plt.show()



"""
	xi_min,xi_max=precession.xi_lim(q,S1,S2)
	Jmin,Jmax=precession.J_lim(q,S1,S2,sep[0])
	#Sso_min,Sso_max=precession.Sso_limits(S1,S2)
	J = np.random.uniform(Jmin,Jmax)
	#St_min,St_max=precession.St_limits(J,q,S1,S2,sep[0])
	xi_low,xi_up=precession.xi_allowed(J,q,S1,S2,sep[0])
	xi=	np.random.uniform(xi_low,xi_up)
	test=(J>=min(precession.J_allowed(xi,q,S1,S2,sep[0])) and J<=max(precession.J_allowed(xi,q,S1,S2,sep[0])))
	print("Is our couple (xi,J) consistent?", test)
	Sb_min,Sb_max=precession.Sb_limits(xi,J,q,S1,S2,sep[0])
	print("S oscillates between\n\tS-=%.3f\n\tS+=%.3f" %(Sb_min,Sb_max))
	S=np.random.uniform(Sb_min,Sb_max)


	print(precession.precession.xi_lim(q,S1,S2),xi)
	print(precession.J_lim(q,S1,S2, sep[0]), J)
	print(precession.St_limits(J,q,S1,S2,sep[0]), S)
	
	#J = np.random.uniform(*precession.J_allowed(xi,q,S1,S2,sep[0]))
	#print(precession.J_allowed(xi,q,S1,S2,sep[0]), J)
	
	#xi = np.random.uniform(*precession.xi_allowed(J,q,S1,S2,sep[0]))
	print(precession.xi_allowed(J,q,S1,S2,sep[0]),xi)
"""
