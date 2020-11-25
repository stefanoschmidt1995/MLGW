import matplotlib.pyplot as plt
import numpy as np
import mlgw.GW_generator as gen #pip install mlgw

import scipy.optimize #for finding a global minimum

class GW_model:
	def __init__(self, signal = None, SNR = 1.):
		"""
		Initialize the model. Takes the parameters [Mc,q] for the signal and the SNR level.
			(I am not completely sure that this is the actual SNR)
		Input:
			signal (2,)		[Mc, q] for the signal in the GW model (if None, no signal is present)
			SNR				signal-to-noise ratio (i.e. variance of the noise is E[signal**2]/SNR)
		"""
		self.t = np.linspace(-4,.05,3000)
		self.model = gen.GW_generator(1)
		if signal is not None:
			self.theta = np.array(signal)
			self.data = self.get_WF(self.theta)
		else:
			self.theta = None
			self.data = np.zeros(self.t.shape) + 1j* np.zeros(self.t.shape)
		np.random.seed(0)
		sigma = ((np.sum(self.data.real**2+self.data.imag**2)+1.)/self.data.shape[0]) /SNR #variance of the noise
		print(np.sqrt(sigma))
		self.data.real = self.data.real +  np.sqrt(sigma) * np.random.normal(0., 1., self.data.shape) #adding noise
		self.data.imag = self.data.imag +  np.sqrt(sigma) * np.random.normal(0., 1., self.data.shape) #adding noise
	
	def __std_theta(self,theta, mchirp = False):
		"Adds to theta = [M,q] the orbital parameters required for the generator. theta[0] can also be given the chirp mass"
		if theta is None:
			return None
		if mchirp:
			#theta = [Mc, q]
			factor = theta[0] * np.power(1. + theta[1], 1.0/5.0)
			m1 = factor * np.power(theta[1], -3.0/5.0)
			m2 = factor * np.power(theta[1], +2.0/5.0)
		else:	
			#theta = [Mtot, q]
			m1 = theta[0]*theta[1]/(1+theta[1])
			m2 = theta[0]/(1+theta[1])
		return [m1,m2,0,0,1e-19,0,0] #[m1,m2,s1,s2,d,iota, phi] values different from m1 and m2 are set to default

	def get_WF(self, theta):
		"Outputs the WF, given orbital parameters theta = [M,q]"
		if theta is None:
			return np.zeros(self.t.shape) + 1j* np.zeros(self.t.shape)
		hp, hc = self.model.get_WF(self.__std_theta(theta), self.t)
		h = (hp +1j*hc)
		return h

	def grad_WF(self, theta):
		"Gradient of the WF evaluated at theta w.r.t. [M,q]"
		grad_Re, grad_Im = self.model.get_grads(self.__std_theta(theta), self.t)
		return (grad_Re[0,:,:2]+1j*grad_Im[0,:,:2])

	def compute_scalar(self, h1,h2):
		"Scalar product between two WFs"
		#as long as no noise is considered, scalar products in TD and FD yield the same results
		assert h1.shape == h2.shape
		if h1.ndim == 1:
			h1 = h1[np.newaxis,:] #(1,D)
			h2 = h2[np.newaxis,:] #(1,D)

		return np.sum(np.multiply(h1,np.conj(h2)), axis = 1).real/h1.shape[1] #(N,)

	def log_likelihood(self,p):
		"Log likelihood of the model evaluated at point p"
		WF = self.get_WF(p)
		#LL = self.compute_scalar(WF,self.data)
		LL = -0.5*self.compute_scalar(WF-self.data,WF-self.data)
		return LL

	def grad_log_likelihood(self,p):
		"Gradient at p of the log-likelihood"
		WF = self.get_WF(p) #(D,)
		grad_WF = self.grad_WF(p) #(D,2)
		#print(WF.shape, grad.shape)
		to_return = np.zeros((grad_WF.shape[1],))
		for i in range(to_return.shape[0]):
			to_return[i] = (self.compute_scalar(grad_WF[:,i],self.data) -self.compute_scalar(grad_WF[:,i],WF))
		return to_return #(2,)

	def E_pot(self,p):
		"Potential energy for optimization"
		return -self.log_likelihood(p)

	def grad_E_pot(self,p):
		"Gradient of potential energy for optimization"
		return -self.grad_log_likelihood(p)


	#generating the signal and the model
minimize = False
noise = 10.
model = GW_model([32.3,2], noise) #signal + noise
#model = GW_model() #noise only model

	#plotting the signal and the data
plt.plot(model.t,model.data.real, label = 'data')
plt.plot(model.t, model.get_WF(model.theta).real, label = 'signal')
try:
	plt.title("Signal @ "+r"$ (\mathcal{M}_c, q) = $"+"({},{})".format(*model.theta)+" SNR = "+str(noise))
except:
	plt.title("Noise")
plt.legend()

#this is for looking for a minimum...
if minimize:
	start = [10,3.4]
	res = scipy.optimize.basinhopping(model.E_pot, start, 1, stepsize = .5, T = .01,
			minimizer_kwargs = {'method':'L-BFGS-B', 'jac':model.grad_E_pot, 'options':{'disp': False}})
	#		,callback=print_fun)
	print(res)


	#creating countor plots
lengrid = 50
x = np.linspace(5,50,lengrid) #chirp mass
y = np.linspace(1,10,lengrid) #mass ratio
X,Y = np.meshgrid(x, y) # grid of point
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
	for j in range(X.shape[1]):
		Z[i,j] = model.log_likelihood([X[i,j], Y[i,j]])
print(x.shape, y.shape)

	#plotting everything
fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 

cp = plt.contourf(X, Y, Z, cmap = "coolwarm") #if you prefer a countor plot
#cp = plt.pcolormesh(X, Y, Z) #if you prefer a colormesh
plt.colorbar(cp)
if minimize:
	plt.plot(*res['x'],'o', ms = 5, c = 'r')

ax.set_title('LL (SNR @ '+str(noise)+')')
ax.set_xlabel(r'$ \mathcal{M}_c(M_\odot)$')
ax.set_ylabel(r'$ q $')
plt.show()





quit()
















