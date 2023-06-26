import sys
sys.path.insert(1, '../../mlgw_v2')
import matplotlib.pyplot as plt
import numpy as np
import GW_generator as gen
from matplotlib import cm

#mchirp = lambda m1, m2: np.power(m1*m2, 3./5.)/np.power(m1+m2, 1./5.) #chirp mass
#eta = lambda m1, m2: np.divide(m1/m2, np.square(1+m1/m2)) #chirp mass

def mchirpeta(m1,m2):
	mchirp = np.power(m1*m2, 3./5.)/np.power(m1+m2, 1./5.) #chirp mass
	eta = np.divide(m1/m2, np.square(1+m1/m2)) #chirp mass
	return mchirp, eta

def m1m2(mchirp, eta):
	q = (1-2*eta +np.sqrt(1-4*eta))/(2*eta)
	M = mchirp* np.power(eta, -3./5.)
	return q*M/(1+q), M/(1+q)

def grad_metric(grad_h_real,grad_h_imag, dt):
	"""
	Gradient of <h,h>; flat PSD.
	Inputs:
		h (N,D,4)
	Outputs:
		grad (N,4,4)
	"""
	return (np.einsum("ijk,ijl->ikl",grad_h_real,grad_h_real) + np.einsum("ijk,ijl->ikl",grad_h_imag,grad_h_imag))*dt
	
def metric(h_real, h_imag, dt):
	"""
	Gradient of <h,h> for the flat PSD case
	Inputs:
		h (N,D,4)
	Outputs:
		metric (N,)
	"""
	return (np.einsum("ij,ij->i", h_real, h_real) + np.einsum("ij,ij->i", h_imag, h_imag))*dt
	

N_wf = 1000
mchirp_eta = True
theta = np.random.uniform([10,10,0,0], [100,100,0.,0], size = (N_wf,4))
	
model = gen.GW_generator(0)#"/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/mlgw_v2/TD_model/model_0")

t = np.linspace(-4,0.1,1000)
h_p,h_c = model.get_modes(theta,t, modes = (2,2), out_type = "realimag") #(N,1000,4)
g_h_real, g_h_imag = model.get_mode_grads(theta,t, modes = (2,2), out_type = "realimag", mchirp_eta = mchirp_eta) #(N,1000,4)

grad_match = grad_metric(g_h_real, g_h_imag, t[1]-t[0]) #(N,2,2)
match = metric(h_p, h_c, t[1]-t[0])

if mchirp_eta:
	var0, var1 = mchirpeta(*theta[:,:2].T) #mchirp, eta values at which grads are evaluated
else:
	var0, var1 =  theta[:,0]+ theta[:,1], np.maximum(theta[:,1]/ theta[:,0] ,theta[:,0]/ theta[:,1] ) #M, q values at which grads are evaluated

n = np.sum(np.square(g_h_real[:,:,:2]), axis =1) #(N,2)
print(n)
n = np.divide(n.T,np.linalg.norm(n, axis = 1)).T

plt.figure()
plt.title("Gradients")
plt.plot(t, g_h_real[0,:,0].T, label = "d/dmchirp")
plt.plot(t, g_h_real[0,:,1].T, label = "d/deta")
plt.plot(t, g_h_real[0,:,2].T, label = "d/ds1")
plt.plot(t, g_h_real[0,:,3].T, label = "d/ds2")
plt.legend()


plt.figure()
plt.title(r"$\partial h / \partial theta$")
plt.quiver(var0, var1, n[:,0], n[:,1])

if mchirp_eta:
	plt.xlabel(r"$\mathcal{M}$")
	plt.ylabel(r"$\eta$")
else:
	plt.xlabel(r"$M$")
	plt.ylabel(r"$q$")
	
plt.figure()
plt.title("Grad metric")
viridis = cm.get_cmap('viridis', 8)

print(grad_match[0,...], match[0])

plt.scatter(var0, var1, c = np.log10(np.linalg.det(grad_match[:,:2,:2])), s = 10)
#plt.scatter(var0, var1, c = np.log10(np.abs(grad_match[:,1,0])), s = 10)
#plt.scatter(var0, var1, c = match, s = 10)
plt.colorbar()

if mchirp_eta:
	plt.xlabel(r"$\mathcal{M}$")
	plt.ylabel(r"$\eta$")
else:
	plt.xlabel(r"$M$")
	plt.ylabel(r"$q$")	
	
	

plt.show()
















