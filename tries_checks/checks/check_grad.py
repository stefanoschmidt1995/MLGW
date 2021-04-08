import sys
sys.path.insert(1, '../../mlgw_v2')
import matplotlib.pyplot as plt
import numpy as np
import GW_generator as gen

###RAW WF WORKS WELL...
###let's see the rest!!

#mchirp = lambda m1, m2: np.power(m1*m2, 3./5.)/np.power(m1+m2, 1./5.) #chirp mass
#eta = lambda m1, m2: np.divide(m1/m2, np.square(1+m1/m2)) #chirp mass

def m1m2(mchirp, eta):
	q = (1-2*eta +np.sqrt(1-4*eta))/(2*eta)
	M = mchirp* np.power(eta, -3./5.)
	return q*M/(1+q), M/(1+q)


model = gen.GW_generator(0)#"/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/mlgw_v2/TD_model/model_0")

mode_name = (2,2)
grad_id = 0
q0= 20

if grad_id == 0:
	mchirp_eta = False
	std_q = lambda q: np.array([[6.*q/7.,q/7.,-0.1,.3,1e-5,2.4234,0.4]]) #total mass
if grad_id == 1:
	mchirp_eta = False
	std_q = lambda q: np.array([[q*20./(1+q),20./(1+q),.1,.3,1e-5,-.3,0.4]]) #q
if grad_id == 2:
	mchirp_eta = False
	std_q = lambda q: np.array([[15,45, q,-.63],[20, 10,.1,.32]]) #spin
if grad_id == 3:
	mchirp_eta = False
	std_q = lambda q: np.array([[15,45, .2,q],[20, 10,.1,.32]]) #spin

if grad_id == 4:
	mchirp_eta = True
	std_q = lambda q: np.array([[*m1m2(q,0.2),.1,.3,1e-5,-.3,0.4]]) #mchirp
if grad_id == 5:
	mchirp_eta = True
	std_q = lambda q: np.array([[*m1m2(20.,q),.1,.3,1e-5,-.3,0.4]]) #eta

#std_q = lambda q: np.array([[3.,-.5, q]])
#std_q = lambda q: np.array([[45,23,-0.1,.3,q,1.4234,0.4]]) #distance
#std_q = lambda q: np.array([[23,53,-0.1,.3,0.2342,q,0.4]]) #iota
#std_q = lambda q: np.array([[45,23,-0.1,.3,0.2342,0.4,q]]) #phi

t = np.linspace(-4,0.05, 5000)
def get_mode(q, mode = mode_name):
	print(std_q(q))
	#amp, ph = model.get_WF(std_q(q), t, plus_cross = False)
	amp, ph = model.get_modes(std_q(q), t, modes = mode_name, out_type = "realimag")
	#h= (hp+1j*hc)
	#amp, ph = model.get_raw_WF(std_q(q))
	return amp[0,:]

def get_grads(q, mode = mode_name):
	#g_amp, g_ph = model.get_mode_obj(mode).get_grads(std_q(q),t, out_type = "realimag")
	g_amp, g_ph = model.get_mode_grads(std_q(q),t, modes = mode_name, out_type = "realimag", mchirp_eta = mchirp_eta)
	print(g_amp.shape)
	#g_amp, g_ph = model.get_raw_grads(std_q(q))
	return g_amp[0,:,grad_id%4]

epsilon = .0000005
WF_4pe = get_mode(q0+epsilon)
WF_4me = get_mode(q0-epsilon)

print(WF_4me.shape, WF_4pe.shape)

#WF_4pe = get_red_coeff(q0+epsilon)
#WF_4me = get_red_coeff(q0-epsilon)

#plt.plot(model.get_time_grid(), (WF_4pe- WF_4me)/(2*epsilon),'o', ms =2, label = "numerical")
#plt.plot(model.get_time_grid(), get_grads(q0) ,'o', ms =2, label = "analytical")
#plt.plot(t, get_WF(q0) ,'o', ms =2, label = "WF")


plt.plot(t,get_grads(q0) ,'o', ms =3, label = "analytical")
plt.plot(t,(WF_4pe- WF_4me)/(2*epsilon),'o', ms =1, label = "numerical")
plt.legend()
plt.show()






