"""
Code to integrate the ODE of PN evolution for orbital angular momentum and spins in a BBH.
Integrates equations (1-4) in https://arxiv.org/abs/1703.03967
See https://arxiv.org/abs/2005.05338 for higher order correction
"""
	#imports
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

	#Basic class for the ODE and its solution
class PN_equations():
	"Class for computing the time evolution of the angular momentum and spins according to PN equations. It implements the equations and solves them with scipy integrator."
	def __init__(self):
		self.M_sun = 4.93e-6 #M_sun in seconds
		return

	def get_q(self,m1,m2):
		"Mass ratio"
		return np.minimum(m1/m2,m2/m1) #wrong convention: q<1

	def get_eta(self,m1,m2):
		"Symmetric mass ratio"
		q = self.get_q(m1,m2)
		return q/np.square(1+q)

	def S_derivative(self,v, L, S1, S2, eta, q):
		"Basic for spin derivative. Called twice by derivative."
		coeff = eta*(2+1.5*q) - 1.5*v *np.sum(np.multiply(q*S1+S2, L), axis = 1) #(N,)
		coeff = coeff * np.power(v,5) #(N,)

		derivative = np.cross(L,S1).T #(3,N)
		derivative = np.multiply(derivative, coeff).T #(N,3)
		derivative = derivative + np.multiply(np.cross(S2,S1).T,0.5*np.power(v,6.)).T

		return derivative #(N,3)

	def L_derivative(self,v, L, S1, S2, m1, m2):
		"Time derivative of orbital angular momentum"
		eta = self.get_eta(m1,m2) #(N,)
		q = self.get_q(m1,m2)	#(N,)
		coeff1 = (2+1.5*q) - 1.5*(v/eta) *np.sum(np.multiply(q*S1+S2, L), axis = 1) #(N,)

		coeff2 = (2+1.5/q) - 1.5*(v/eta) *np.sum(np.multiply(S2/q+S1, L), axis = 1) #(N,)

		derivative = np.multiply(np.cross(S1,L).T, np.multiply(coeff1,np.power(v,6))).T
		derivative += np.multiply(np.cross(S2,L).T, np.multiply(coeff2,np.power(v,6))).T
		return derivative

	def v_derivative(self,v, L, S1, S2, m1, m2):
		"Time derivative of orbital frequency v"
		eta = self.get_eta(m1,m2) #(N,)
		q = self.get_q(m1,m2)	#(N,)

		#there is the issue of units for M!!! What are they??

		#return np.zeros(v.shape) #DEBUG: switching off radiation reaction

			#for a,b coefficients, see Appendix A of: https://arxiv.org/abs/1307.4418
		#betas and sigmas
		scalar = lambda x,y: np.sum(np.multiply(x,y), axis =1) #(N,3) x (N,3) -> (N,)
		beta_3 = (113./12.+ 25./4.* m1/m2* scalar(S2,L) ) + (113./12.+ 25./4.* m2/m1* scalar(S1,L) )
		beta_3 = np.divide(beta_3, np.square(m1+m2))

		beta_5 = ( ((31319./1008. - 1159./24.*eta) + m2/m1 *(809./84. - 281./8. *eta) ) * scalar(S1,L) ) + ( ((31319./1008. - 1159./24.*eta) + m1/m2 *(809./84. - 281./8. *eta) )* scalar(S2,L) )
		beta_5 = np.divide(beta_5, np.square(m1+m2))

		beta_6 = (75./2.+ 151./6.* m1/m2* scalar(S2,L) ) + (75./2.+ 151./6.* m2/m1* scalar(S1,L) )
		beta_6 = np.divide(beta_6, np.pi*np.square(m1+m2))

		beta_7 = ( ((130325./756. - 796069./2016.*eta + 100019./864. * eta**2) + m2/m1 *(1195759./18144. - 257023./1008.*eta + 2903./32. * eta**2) ) * scalar(S1,L) ) + ( ((130325./756. - 796069./2016.*eta + 100019./864. * eta**2) + m1/m2 *(1195759./18144. - 257023./1008.*eta + 2903./32. * eta**2) ) * scalar(S2,L) )
		beta_7 = np.divide(beta_7, np.square(m1+m2))

		sigma_4 = 247./48. * scalar(S1,S2) - 721./48. * scalar(S1,L) * scalar(S2,L)
		sigma_4 = np.divide(sigma_4, m1*m2*np.square(m1+m2))
		sigma_4 += np.divide( 233./96. * scalar(S1,S1) - 719./96.* np.square(scalar(S1,L)), np.square(m1+m2)*m1**2)
		sigma_4 += np.divide( 233./96. * scalar(S2,S2) - 719./96.* np.square(scalar(S2,L)), np.square(m1+m2)*m2**2)

		#a coefficients
		a_0 = 96./5.*eta
		a_2 = -743./336. -11./4.*eta
		a_3 = 4 * np.pi - beta_3
		a_4 = 34103./18144. + 13661./2016. * eta + 59./18.*np.square(eta) - sigma_4
		a_5 = - 4159./672. *np.pi - 189./8. * np.pi * eta - beta_5
		a_6 = 16447322263./139708800. + 16/3.*np.pi - 856./105.*np.log(16) - 1712./105. * np.euler_gamma - beta_6 +	eta * (451./48.*np.pi**2 - 56198689./217728.) + np.square(eta) * 541./896. + np.power(eta, 3)* 5605./2592.
		a_7 = - 4415./4032. *np.pi + 358675./6048. *np.pi*eta + 91495./1512. *np.pi*np.square(eta) - beta_7

		#b coefficients
		b_6 = -1712./315.

		g_n = [1./a_0, 0., -a_2/a_0, -a_3/a_0, -(a_4- a_2**2)/a_0, -(a_5- 2*a_2*a_3)/a_0, 
				-(a_6- 2*a_4*a_4- a_3**2 + a_2**2)/a_0, -(a_7- 2*a_5*a_2- 2*a_4*a_3 + 3.*a_3*a_2**2)/a_0 ]
		g_nl = [0. for i in range(8)]
		g_nl[6] = -3*b_6/a_0

		derivative =  0.		
		for i in range(8): #loop on g_n
			derivative += np.multiply(g_n[i]+3.*g_nl[i]*np.log(v), np.power(v,i))

		derivative = np.multiply(np.power(derivative,-1), np.power(v,9)/3.)
		return derivative


	def __derivative(self, v,L,S1, S2, m1, m2):
		"""
		Derivative of the evolution of the BBH angular momentum and spins, computed according to https://arxiv.org/abs/1703.03967
		Assumes 2D inputs
		Inputs:
			v (N,)		Orbital frequency (kind of)
			L (N,3)		Orbital angular momentum
			S1 (N,3)	Spin 1
			S2 (N,3)	Spin 2
			m1 (N,)		First BH mass
			m2 (N,)		Second BH mass
		Outputs:
			v_dot (N,)		time derivative of v
			L_dot (N,3)		time derivative of L
			S1_dot (N,3)		time derivative of L
			S2_dot (N,3)		time derivative of L
		"""
		v_dot = self.v_derivative(v, L, S1, S2, m1, m2) #(N,)

		S1_dot = self.S_derivative(v, L, S1, S2, self.get_eta(m1,m2), self.get_q(m1,m2)) #(N,3)
		S2_dot = self.S_derivative(v, L, S2, S1, self.get_eta(m1,m2), 1./self.get_q(m1,m2)) #(N,3)

		L_dot = -(S1_dot+S2_dot)
		#L_dot = self.L_derivative(v, L, S1, S2, m1, m2) #(N,3)

		L_abs = self.get_eta(m1,m2)/v #(N,)/()
		#print(L_dot*L_abs+S1_dot+S2_dot, np.linalg.norm(L),v)

		return v_dot, L_dot, S1_dot, S2_dot



	def derivative(self, v,L,S1, S2, m1, m2):
		"""
		Derivative of the evolution of the BBH angular momentum and spins, computed according to https://arxiv.org/abs/1703.03967
		Assumes 1D/2D inputs; it calls __derivative
		Inputs:
			v (N,)/()		Orbital frequency (kind of)
			L (N,3)/(3,)	Orbital angular momentum
			S1 (N,3)/(3,)	Spin 1
			S2 (N,3)/(3,)	Spin 2
			m1 (N,)/()		First BH mass
			m2 (N,)/()		Second BH mass
		Outputs:
			v_dot (N,)/()			time derivative of v
			L_dot (N,3)/(3,)		time derivative of L
			S1_dot (N,3)/(3,)		time derivative of L
			S2_dot (N,3)/(3,)		time derivative of L

		"""
		if L.shape != S1.shape and S2.shape != S1.shape:
			raise RuntimeError("Wrong input shapes. Inputs spins and L must be (None,3)")
		v = np.squeeze(np.array(v))
		L = np.array(L)
		S1 = np.array(S1)
		S2 = np.array(S2)

		if L.ndim == 1:
			v = v[None]
			L = L[None,:]
			S1 = S1[None,:]
			S2 = S2[None,:]
			v_dot, L_dot, S1_dot, S2_dot = self.__derivative(v,L,S1, S2, m1, m2)
			return v_dot[0], L_dot[0,:], S1_dot[0,:], S2_dot[0,:]
		else:
			return self.__derivative(v,L,S1, S2, m1, m2)


	def scipy_derivative(self, t, X, m1, m2):
		"""
		Mask for scipy of function self.derivative of X = [v, L_n, S_1, S_2] w.r.t. time.
		Inputs:
			t 				Dummy variable (for scipy)
			X (N,10)/(10,)	Independent variable: [v (1,), L_n (3,), S_1 (3,), S_2 (3,)]
			m1 (N,)/()		First BH mass
			m2 (N,)/()		Second BH mass
		Output:
			derivative (N,10)/(10,)		derivative of the quantities
		"""
		X = np.array(X)
		if X.ndim == 1:
			X = X[None,:]
			to_reshape = True
		else:
			to_reshape = False

		v = X[:,0] #(N,)
		L = X[:,1:4]	#(N,3)
		S1 = X[:,4:7]	#(N,3)
		S2 = X[:,7:]	#(N,3)
		
#		v_dot, L_dot, S1_dot, S2_dot = self.__derivative(v,L,S1, S2, m1*self.M_sun, m2*self.M_sun)
		v_dot, L_dot, S1_dot, S2_dot = self.__derivative(v,L,S1, S2, m1, m2)

		res = np.concatenate([v_dot[:,None], L_dot, S1_dot, S2_dot], axis = 1) #(N,10)

		if to_reshape:
			return res[0,:]
		else:
			return res

	def get_L(self,v,L, m1, m2):
		"Given orbital frequency and L directions, it computes the coordinates of L."
		L_abs = self.get_eta(m1,m2)/v #(N,)/()
		L = np.multiply(L.T, L_abs).T

		return L

	def get_L_versor(self,theta_L, phi_L):
		"Computes the direction of L, given the angles"
		if isinstance(theta_L, float):
			L = np.zeros((1,3))			
		else:
			L = np.zeros((len(theta_L),3))
		L[:,0] = np.sin(theta_L)*np.cos(phi_L)
		L[:,1] = np.sin(theta_L)*np.sin(phi_L)
		L[:,2] = np.cos(theta_L)

		return np.squeeze(L)

	def get_v(self,f):
		"Computes v, given the orbital frequency in Hertz"
		return np.power(f*self.M_sun, 1./3.)
		

	def get_cos_theta_JL(self, X, m1, m2):
		"Computes the total angular momentum"
		v = X[:,0] #(N,)
		L = X[:,1:4]	#(N,3)
		S1 = X[:,4:7]	#(N,3)
		S2 = X[:,7:]	#(N,3)

		L_abs = self.get_eta(m1,m2)/v #(N,)/()
		S_sq = np.square(np.linalg.norm(S1+S2, axis =1))
		L_sq = np.square(L_abs)
		J_sq = np.square(np.linalg.norm(self.get_J(X,m1,m2), axis = 1))

		cos_theta_JL = 0.5*(J_sq+L_sq-S_sq)/np.sqrt(J_sq*L_sq)
		return cos_theta_JL

	def get_J(self, X, m1, m2):
		"Computes the total angular momentum"
		v = X[:,0] #(N,)
		L = X[:,1:4]	#(N,3)
		S1 = X[:,4:7]	#(N,3)
		S2 = X[:,7:]	#(N,3)
		L = np.multiply(L.T, self.get_eta(m1,m2)/v).T
		return L + S1 + S2

	def get_cos(self,A,B):
		"Computes the cosinus between two vector"
		
		return np.sum(np.multiply(np.divide(A.T,np.linalg.norm(A,axis=1)).T, np.divide(B.T,np.linalg.norm(B,axis=1)).T), axis =1 ) #(N,)

if __name__ == "main":

	eqs = PN_equations()

	m1, m2 = 0.6, 0.4


	v_0 = eqs.get_v(10.)
	print(v_0)
		#dimensionless spins
	S1 = np.array([0.,0.,0.1]) 
	S2 = -np.array([0., -0., 0.1])
	#initial_cond = [v_0, *eqs.get_L_versor(np.pi/2., 0.5), *(S1*m1**2), *(S2*m2**2)]
	initial_cond = [v_0, 0,0,1., *(S1*m1**2), *(S2*m2**2)]
	#print(eqs.get_J(np.array([initial_cond]),m1,m2))

	times = np.linspace(0., 1.1e9, 10000)
	#sol, infodict = scipy.integrate.odeint(eqs.scipy_derivative, np.array(initial_cond), times, args=(m1, m2), tfirst = True, full_output = True)
	res = scipy.integrate.solve_ivp(eqs.scipy_derivative,  (times[0], times[-1]), np.array(initial_cond), t_eval = times, method ="BDF", args=(m1, m2))	
	sol = res['y'].T

	L_n = sol[:,1:4]	#(N,3)
	L = eqs.get_L(sol[:,0],L_n,m1,m2)
	S1 = sol[:,4:7]	#(N,3)
	S2 = sol[:,7:]	#(N,3)

		#orthonormal basis
	J_n = np.divide(eqs.get_J(sol,m1,m2).T,np.linalg.norm(eqs.get_J(sol,m1,m2), axis = 1)).T
	J_n_orth = np.cross(J_n[0,:], np.array([0.,0.,1.]))
	J_n_orth = J_n_orth/np.linalg.norm(J_n_orth)
	J_n_orth_2 = np.cross(J_n, J_n_orth)
	print(J_n)



	plt.figure()
	plt.ylabel(r"$\phi$")
	cos_phi = eqs.get_cos(L,np.repeat([J_n_orth], L.shape[0], axis = 0))
	phi = np.cumsum(np.arccos(cos_phi))
	plt.plot(times*eqs.M_sun, phi)

	plt.figure()
	plt.ylabel(r"$\cos(\theta_{JL})$")
	#plt.plot(times*eqs.M_sun, L[:,0]) #to scale total mass

	plt.plot(times*eqs.M_sun, eqs.get_cos(J_n,L) , label = "scalar product")
	plt.plot(times*eqs.M_sun, (eqs.get_cos_theta_JL(sol, m1, m2)), label = "analytical")


	plt.legend()

	plt.figure()
	plt.xlabel("L_x")
	plt.ylabel("L_y")
	plt.xlim([-1,1])
	plt.ylim([-1,1])
	plt.plot(sol[:,1], sol[:,3])

	#plt.show()

	plt.figure()
	plt.ylabel(r"$J_{tot}$")
	plt.plot(times*eqs.M_sun, np.linalg.norm(eqs.get_J(sol,m1,m2), axis = 1)) #norm
	#plt.plot(times*eqs.M_sun, eqs.get_J(sol,m1,m2)[:,2]) #norm

	plt.show()







		
	
		





		
	

