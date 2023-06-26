import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(1, '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/routines') 

import corner
import GW_generator as gen
import scipy.optimize
from shgo import shgo	

def compute_optimal_overlap(h1,h2):
	#print(h1.shape, h2.shape)
	assert h1.shape == h2.shape
	if h1.ndim == 1:
		h1 = h1[np.newaxis,:] #(1,D)
		h2 = h2[np.newaxis,:] #(1,D)

	scalar = lambda h1_, h2_: np.sum(np.multiply(h1_,np.conj(h2_)), axis = 1)/h1_.shape[1] #(N,)

	norm_factor = 1.#np.sqrt(np.multiply(scalar(h1,h1).real, scalar(h2,h2).real))
	overlap = scalar(h1,h2) #(N,)
	phi_optimal = np.angle(overlap) #(N,) #this is not a good idea for the gradients... Find an alternative
	overlap = np.divide(scalar(h1,h2*np.exp(1j*phi_optimal)).T, norm_factor).T
	overlap = overlap.real

	return overlap#, phi_optimal


class GW_model:
	def __init__(self):
		self.dim=7
		self.prior_bounds = np.array([[5, 25],[1,10], [-0.8,0.8], [-0.8,0.8], [np.log(1e-21), np.log(1e-19)], [0,3.14], [0,2.28] ])
		self.t = np.linspace(-4,.05,3000)
		self.model = gen.GW_generator("/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/definitive_code/TD_model_TEOBResumS")
		self.q = np.array([10, 5, 0.3, -0.4, np.log(1e-2), 0, 0])
		self.data = self.get_WF(self.q)
		np.random.seed(0)
		self.data = self.data* np.random.normal(1, 5., self.data.shape) #adding noise
	
	def std_q(self,q):
		#q = [Mc, q, s1, s2, d, iota, phi]
		#assert len(q)==self.dim
		factor = q[0] * np.power(1. + q[1], 1.0/5.0)
		m1 = factor * np.power(q[1], -3.0/5.0)
		m2 = factor * np.power(q[1], +2.0/5.0)

		#return [q[1]*q[0]/(1+q[1]),q[0]/(1+q[1]),q[2],q[3], np.exp(q[4]), q[5], q[6]]
		return [m1,m2,q[2],q[3], np.exp(q[4]), q[5], q[6]]
		#return [q[1]*q[0]/(1+q[1]),q[0]/(1+q[1]),q[2],q[3], np.exp(self.q[4]), self.q[5], self.q[6]]
		#return [q[1]*q[0]/(1+q[1]),q[0]/(1+q[1]),self.q[2],self.q[3], np.exp(self.q[4]), self.q[5], self.q[6]]
		#return [q[0]*self.q[0]/(1+q[0]),self.q[0]/(1+q[0]),q[1],self.q[3], np.exp(self.q[4]), self.q[5], self.q[6]]

	def get_WF(self, q):
		hp, hc = self.model.get_WF(self.std_q(q), self.t)
		h = (hp +1j*hc)
		return h[0,:]

	def grad_WF(self, q):
		grad_Re, grad_Im = self.model.get_grads(self.std_q(q), self.t)
		return (grad_Re[0,:,:]+1j*grad_Im[0,:,:])

	def new_point(self):
		return np.array([np.random.uniform(self.prior_bounds[i,0],self.prior_bounds[i,1]) for i in range(self.dim)])
			
	def log_likelihood(self,p):
		#print(self.std_q(p))
		WF = self.get_WF(p)
		#print(p.shape, WF.shape)
		LL = 1e34*compute_optimal_overlap(WF,self.data)
		#LL = LL- 0.5 *compute_optimal_overlap(WF,WF) #adding the rest (useless)
		#LL = LL - 0.5 *compute_optimal_overlap(self.data,self.data) 
		return LL
		#return -0.5*compute_optimal_overlap(WF-self.data,WF-self.data)
		
	def log_prior(self, p):
		return 0
		if not isinstance(p, np.ndarray):
			p = np.array(p)
			#check boundaries
		up = self.prior_bounds[:,1] - p 
		down = p - self.prior_bounds[:,0]
		#print(p, up, down)
		up = up[:4]
		down = down[:4]
		if np.all(up>=0) and np.all(down>=0):
			logP =0 
		else:
			logP = -1e50
		return logP
		
	def potential_energy(self, p):
		E_pot = -(self.log_prior(p)+self.log_likelihood(p))
		#print(E_pot, p)
		#quit()
		return E_pot
		
	def potential_energy_gradient(self,p):
		grad = self.grad_WF(p)
		WF = self.get_WF(p)
		#print(WF.shape, grad.shape)
		to_return = np.zeros((grad.shape[1],))
		for i in range(to_return.shape[0]):
			to_return[i] = -(compute_optimal_overlap(grad[:,i],self.data) -compute_optimal_overlap(grad[:,i],WF))
		#print(to_return.shape)
		return to_return

	def p_pot(self, p):
		return np.exp(-self.potential_energy(p))

	def grad_p_pot(self, p):
		return np.exp(-self.potential_energy(p))* self.potential_energy_gradient(p)[[0,1]]



model = GW_model()

#print(model.grad_p_pot([30, 6]))

#print("Bounds", *model.prior_bounds)
#print("True: ", model.q, model.p_pot(model.q))

def print_fun(x, f, accepted):
	print(x[:4])
	print("at minimum %.4f accepted %d" % (f, int(accepted)))


start = model.new_point()

#res = scipy.optimize.basinhopping(model.potential_energy, start, 10, stepsize = .5, T = .01,
#		minimizer_kwargs = {'method':'L-BFGS-B', 'jac':model.potential_energy_gradient, 'options':{'disp': False}},
#		callback=print_fun)

bounds = [*model.prior_bounds]
print(bounds)
#result = shgo(model.p_pot,  bounds)
#print(result)


#res = scipy.optimize.shgo(model.p_pot, bounds = [*model.prior_bounds],
#		options = {'f_min': -1, 'jac': model.grad_p_pot, 'disp': True})


#res = scipy.optimize.minimize(model.potential_energy, start, method = 'Nested-',
		#jac = model.potential_energy_gradient,
		#constraints = scipy.optimize.LinearConstraint(np.identity(7), lb = model.prior_bounds[:,0], ub = model.prior_bounds[:,1]),
		#bounds = [*model.prior_bounds],
		#options = {'disp': True})
#print(res)
#print("Start value: ", start, model.potential_energy(start))



from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

 
x = np.arange(5,25,.5)
#x = np.arange(-0.8,0.8,.05)
y = np.arange(1,10,0.1)
X,Y = np.meshgrid(x, y) # grid of point
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
	for j in range(X.shape[1]):
		#Z[i,j] = model.p_pot([X[i,j], Y[i,j]]) # evaluation of the function on the grid
		#Z[i,j] = model.potential_energy([X[i,j], Y[i,j], 0.3, -0.4, np.log(1e-2), 0, 0])
		Z[i,j] = model.p_pot([X[i,j], Y[i,j], 0.3, -0.4, np.log(1e-2), 0, 0])
		#Z[i,j] = model.potential_energy([10, Y[i,j], X[i,j], -0.4, np.log(100e-20), 0, 0]) 
print(x.shape, y.shape)
im = imshow(Z,cmap=cm.RdBu) # drawing the function
	# adding the Contour lines with labels
#cset = contour(Z,np.arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
#clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
plt.xticks(range(0,len(x),10),x[range(0,len(x),10)])
plt.yticks(range(0,len(y),3),y[range(0,len(y),3)])
colorbar(im) # adding the colobar on the right
show()





quit()
















